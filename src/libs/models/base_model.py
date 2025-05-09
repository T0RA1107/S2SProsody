from collections import OrderedDict
from logging import getLogger
import torch

from . import networks

logger = getLogger(__name__)


class BaseModel:
    def __init__(self, args, preprocess_config, model_config, train_config, isTrain=True):
        self.args = args
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        self.speaker_num = model_config["speaker_num"]
        self.isTrain = isTrain
        self.local_rank = args.local_rank
        self.dist = args.ngpus > 1
        self.device = torch.device(f"cuda:{args.local_rank}") if args.ngpus > 0 else torch.device("cpu")  # get device name: CPU or GPU
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []

    def setup(self, train_config, train_data_size=-1):
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(self.optimizers[0], train_config, train_data_size),
                networks.get_scheduler(self.optimizers[1], train_config, train_data_size, 0, train_config["optimizer"]["discriminator_epoch"] * train_data_size),
            ]

    def set_train_mode(self):
        """Make models train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.train()

    def set_eval_mode(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, "loss_" + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and \
                    (key == "running_mean" or key == "running_var"):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and \
               (key == "num_batches_tracked"):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, save_path):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name "%s_net_%s.pth" % (epoch, name)
        """
        ckpt = torch.load(save_path)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = ckpt[name]
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)
                net.to(self.device)

    def save_networks(self, save_path):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name "%s_net_%s.pth" % (epoch, name)
        """
        param_dict = dict()
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)

                if self.dist:
                    param_dict[name] = net.module.state_dict()
                else:
                    param_dict[name] = net.cpu().state_dict()
                    net.to(self.device)
        torch.save(param_dict, save_path)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        logger.debug("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    logger.debug(net)
                logger.debug("[Network %s] Total number of parameters : %.3f M" % (name, num_params / 1e6))
        logger.debug("-----------------------------------------------")

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for i, (name, scheduler) in enumerate(zip(self.model_names, self.schedulers)):
            old_lr = self.optimizers[i].param_groups[0]["lr"]
            if scheduler.T_0 <= scheduler.last_epoch:
                continue
            if self.train_config["GAN"]["lr_policy"] == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()
            lr = self.optimizers[i].param_groups[0]["lr"]

    def get_learning_rate(self):
        lr_dict = dict()
        for i, name in enumerate(self.model_names):
            lr = self.optimizers[i].param_groups[0]["lr"]
            lr_dict["lr/" + name] = lr
        return lr_dict

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
