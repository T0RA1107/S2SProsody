import torch
import torch.nn as nn
import os
from collections import OrderedDict
import itertools
import numpy as np

from util.audio_pool import AudioPool
from . import networks
from .sign2audio import Sign2Speech
from .prosody_estimator import ProsodyEstimator1D, ProsodyEstimator2D, ProsodyDistEstimator1D

from FastSpeech2.model.loss import FastSpeech2Loss


def make_mask_from_lens(length, max_length=None):
    if max_length is None:
        max_length = torch.max(length)
    bs = length.shape[0]
    idxs = torch.arange(0, max_length, device=length.device).unsqueeze(0).repeat(bs, 1)
    mask = idxs < length.unsqueeze(1).repeat(1, max_length)
    return mask


class BaseModel:
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, args, preprocess_config, model_config, train_config, isTrain=True):
        """Initialize the BaseModel class.

        Parameters:
            args-- stores all the experiment flags;

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.args = args
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        self.speaker_num = model_config["speaker_num"]
        self.gpu_ids = train_config["gpu_ids"]
        self.isTrain = isTrain
        self.local_rank = args.local_rank
        self.dist = args.ngpus > 1
        self.device = torch.device(f"cuda:{args.local_rank}") if self.gpu_ids else torch.device("cpu")  # get device name: CPU or GPU
        self.save_dir = train_config["path"]["ckpt_path"]  # save all the checkpoints to save_dir
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []


    def setup(self, train_config):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, train_config) for optimizer in self.optimizers]
        # if not self.isTrain or args.continue_train:
        #     load_suffix = "iter_%d" % args.load_iter if args.load_iter > 0 else args.epoch
        #     self.load_networks(load_suffix)
        # self.print_networks(args.verbose)

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
        ckpt = torch.loat(save_path)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = ckpt[name]
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

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
        torch.save(param_dict, save_path)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print("[Network %s] Total number of parameters : %.3f M" % (name, num_params / 1e6))
        print("-----------------------------------------------")

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for i, (name, scheduler) in enumerate(zip(self.model_names, self.schedulers)):
            if name == "Prosody_estimator": continue
            old_lr = self.optimizers[i].param_groups[0]["lr"]
            if self.train_config["GAN"]["lr_policy"] == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()
            lr = self.optimizers[i].param_groups[0]["lr"]

            # if self.local_rank == 0:
            #     print(f"{name}: learning rate {old_lr:.7f} -> {lr:.7f}")

    def get_learning_rate(self):
        lr_dict = dict()
        for i, name in enumerate(self.model_names):
            if name == "Prosody_estimator": continue
            lr = self.optimizers[i].param_groups[0]["lr"]
            lr_dict[name] = lr
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


class SemiCycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires "--dataset_mode unaligned" dataset.
    By default, it uses a "--netG resnet_9blocks" ResNet generator,
    a "--netD basic" discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ("--gan_mode lsgan").

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_sign, lambda_audio, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_sign * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_audio * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_audio + ||G_B(A) - A|| * lambda_sign) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument("--lambda_sign", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_audio", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")

        return parser

    def __init__(self, args, preprocess_config, model_config, train_config, configs_ft=None, isTrain=True, distributed=False):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, args, preprocess_config, model_config, train_config, isTrain)
        self.prosody_dist = train_config["loss"]["prosody"]["dist"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_sign2audio = Sign2Speech(preprocess_config, model_config)
        self.netG_sign2audio = Sign2Speech(preprocess_config, model_config)
        self.model_names.append("G_sign2audio")
        self.target_speaker = model_config["speaker"]["female"]
        self.netG_sign2audio = networks.init_net(self.netG_sign2audio, args, distributed=distributed, gpu_ids=self.gpu_ids)
        if configs_ft is not None:
            train_config_ft = configs_ft[2]
            ckpt_path = os.path.join(
                train_config_ft["path"]["ckpt_path"],
                "{}.pth.tar".format(args.restore_step_ft),
            )
            if args.local_rank == 0:
                print(f"Load {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            if isinstance(self.netG_sign2audio, torch.nn.parallel.DistributedDataParallel):
                self.netG_sign2audio.module.load_state_dict(ckpt["model"], strict=False)
            else:
                self.netG_sign2audio.load_state_dict(ckpt["model"], strict=False)

        self.audio2sign = None

        self.step = 0

        if self.isTrain:  # define discriminators
            self.netD_audio = networks.define_D(
                model_config["D_audio"]["input_nc"], model_config["D_audio"]["ndf"], "audio",
                model_config["D_audio"]["n_layers_D"], model_config["D_audio"]["norm"],
                model_config["D_audio"]["init_type"], model_config["D_audio"]["init_gain"],
                args, distributed, train_config["gpu_ids"])
            self.model_names.append("D_audio")
            self.loss_log_D = {}
            self.weight_prosody = train_config["loss"]["weight"]["prosody"]

            self.netProsody_estimator = ProsodyDistEstimator1D(
                3, train_config["loss"]["prosody"]["bins"], 4
            )
            self.netProsody_estimator = networks.init_net(self.netProsody_estimator, args, distributed=distributed, gpu_ids=self.gpu_ids)
            self.model_names.append("Prosody_estimator")

            self.mean = train_config["discriminator"]["noise"]["mean"]
            self.std = train_config["discriminator"]["noise"]["std"]
            self.fake_audio_pool = AudioPool(train_config["GAN"]["pool_size"])  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(train_config["GAN"]["gan_mode"]).to(self.device, non_blocking=True)  # define GAN loss.
            # self.criterionProsody = nn.MSELoss(reduction="none").to(self.device, non_blocking=True)
            self.criterionProsody = nn.CrossEntropyLoss().to(self.device, non_blocking=True)
            # train_parameters = list(self.netG_sign2audio.parameters())
            train_parameters = []
            train_layers = ["sign_processer", "s2s_mixier", "visual_project"]
            for name, param in self.netG_sign2audio.named_parameters():
                if any(layer_name in name for layer_name in train_layers):
                    train_parameters.append(param)
            train_parameters += list(self.netProsody_estimator.parameters())

            self.optimizer_G = torch.optim.Adam(train_parameters,
                                                lr=train_config["optimizer"]["lr_G_s2a"],
                                                betas=train_config["optimizer"]["betas"])
            self.optimizer_D = torch.optim.SGD(self.netD_audio.parameters(),
                                                lr=train_config["optimizer"]["lr_D_a"],
                                                momentum=0.9)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, inputs):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            inputs (dict): include the data itself and its metadata information.
        """
        self.real_sign = inputs["sign"]
        self.fake_raw_texts = self.real_sign.raw_texts
        self.fake_text_lens = self.real_sign.token_length
        self.prosody_label = self.real_sign.prosody_label.to(self.device, non_blocking=True)

        self.real_audio = inputs["audio"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        batch_size = self.real_sign.text_tokens.shape[0]
        token_length = self.real_sign.token_length.to(self.device, non_blocking=True)
        max_src_len = token_length.max().to(self.device, non_blocking=True)
        text_tokens = self.real_sign.text_tokens[:, :max_src_len].to(self.device, non_blocking=True)

        speakers = torch.full((batch_size,), self.target_speaker, device=self.device).long()

        # without sign language TTS
        with torch.no_grad():
            pred = self.netG_sign2audio(
                speakers,
                text_tokens,
                token_length,
                max_src_len,
            )

            self.synth_mel_masks = make_mask_from_lens(pred.mel_lens,
                                                       max_length=pred.postnet_output.shape[1])
            self.synth_audio = pred.postnet_output.masked_fill(
                pred.mel_masks.unsqueeze(2).repeat(1, 1, pred.postnet_output.shape[2]), 0.0).unsqueeze(1)

            self.synth_audio_lens = pred.mel_lens.detach().cpu()
            prosody_predictions = torch.cat([
                pred.p_predictions.unsqueeze(1),
                pred.e_predictions.unsqueeze(1),
                pred.log_d_predictions.unsqueeze(1)
            ], dim=1)
            self.pred_prosody_label_wo_sign = self.netProsody_estimator(prosody_predictions)

        # with sign language TTS
        pred = self.netG_sign2audio(
            speakers,
            text_tokens,
            token_length,
            max_src_len,
            key_point=self.real_sign.visual_prefix.to(self.device, non_blocking=True)
        )

        self.fake_mel_masks = make_mask_from_lens(pred.mel_lens, max_length=pred.postnet_output.shape[1])
        self.fake_audio_with_sign = pred.postnet_output.masked_fill(
            pred.mel_masks.unsqueeze(2).repeat(1, 1, pred.postnet_output.shape[2]), 0.0).unsqueeze(1)

        self.fake_audio_with_sign_lens = pred.mel_lens
        self.speakers = speakers.detach().cpu()
        prosody_predictions = torch.cat([
            pred.p_predictions.unsqueeze(1),
            pred.e_predictions.unsqueeze(1),
            pred.log_d_predictions.unsqueeze(1)
        ], dim=1)
        self.pred_prosody_label = self.netProsody_estimator(prosody_predictions)

        torch.cuda.empty_cache()

    def augmentation_audio(self, audio, mask):
        return audio
        noise = torch.randn_like(audio).to(self.device) * self.std + self.mean
        noise = noise * mask.unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, audio.shape[3])
        return audio + noise

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(self.augmentation_audio(real, self.synth_mel_masks))
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(self.augmentation_audio(fake.detach(), self.fake_mel_masks))
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        loss_D = loss_D.detach().cpu()
        torch.cuda.empty_cache()
        return loss_D

    def backward_D_audio(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_audio, fake_audio_lens = self.fake_audio_pool.query(
            self.fake_audio_with_sign.detach().cpu(),
            self.fake_audio_with_sign_lens.detach().cpu()
            )
        fake_audio = fake_audio.to(self.device, non_blocking=True)

        real_audio = self.real_audio.mels.float().to(self.device, non_blocking=True)
        real_audio_lens = self.real_audio.mel_lens
        input_len = min(min(fake_audio_lens), min(real_audio_lens))
        loss_D = self.backward_D_basic(
            self.netD_audio,
            real_audio[:, :input_len].unsqueeze(1),
            fake_audio[:, :input_len].unsqueeze(1)
        )
        torch.cuda.empty_cache()
        return { "GAN loss/D": loss_D }

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        loss_log = dict()

        # GAN loss D_audio(G_sign2audio(sign))
        fake_audio = self.fake_audio_with_sign
        pred_fake = self.netD_audio(self.augmentation_audio(fake_audio, self.fake_mel_masks))
        loss_G_audio = self.criterionGAN(pred_fake, True)
        loss_log["GAN loss/G"] = loss_G_audio.detach().cpu()

        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_prosody = [self.criterionProsody(self.pred_prosody_label[i], self.prosody_label[:, i]) for i in range(len(self.pred_prosody_label))]
        loss_log["prosody loss/v_loss"] = sum(loss_prosody[:2]).detach().cpu() / 2.
        loss_log["prosody loss/a_loss"] = sum(loss_prosody[2:4]).detach().cpu() / 2.

        loss_prosody = sum(loss_prosody) / 4.
        loss_log["prosody loss/total"] = loss_prosody.detach().cpu()

        with torch.no_grad():
            loss_prosody_wo_sign = [self.criterionProsody(self.pred_prosody_label_wo_sign[i], self.prosody_label[:, i]) for i in range(len(self.pred_prosody_label_wo_sign))]
            loss_log["prosody loss without sign/v_loss"] = sum(loss_prosody_wo_sign[:2]).detach().cpu() / 2.
            loss_log["prosody loss without sign/a_loss"] = sum(loss_prosody_wo_sign[2:4]).detach().cpu() / 2.
            loss_log["prosody loss without sign/total"] = sum(loss_prosody_wo_sign).detach().cpu() / 4.
        # combined loss and calculate gradients
        loss_G = loss_G_audio
        loss_G += loss_prosody * self.weight_prosody

        loss_G.backward()

        torch.cuda.empty_cache()
        return loss_log

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss_log = dict()
        self.set_train_mode()
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_audio], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad(set_to_none=True)  # set G"s gradients to zero
        loss_log_G = self.backward_G()             # calculate gradients for G
        loss_log.update(loss_log_G)
        self.optimizer_G.step()       # update G"s weights
        if self.step == 0:
            # D_A and D_B
            self.set_requires_grad([self.netD_audio], True)
            self.optimizer_D.zero_grad(set_to_none=True)   # set D"s gradients to zero
            self.loss_log_D = self.backward_D_audio()      # calculate gradients for D_audio
            self.optimizer_D.step()  # update D_A and D_B"s weights
        self.step = (self.step + 1) % 2
        loss_log.update(self.loss_log_D)
        torch.cuda.empty_cache()
        return loss_log
