import torch
import os
from collections import OrderedDict
import itertools

from util.audio_pool import AudioPool
from . import networks
from .sign2audio import Sign2Speech, FastSpeech2

from FastSpeech2.utils.tools import to_device


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
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = train_config["path"]["ckpt_path"]  # save all the checkpoints to save_dir
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'


    def setup(self, train_config):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, train_config) for optimizer in self.optimizers]
        # if not self.isTrain or args.continue_train:
        #     load_suffix = 'iter_%d' % args.load_iter if args.load_iter > 0 else args.epoch
        #     self.load_networks(load_suffix)
        # self.print_networks(args.verbose)

    def set_train_mode(self):
        """Make models train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()


    def set_eval_mode(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def set_test_mode(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for i, (name, scheduler) in enumerate(zip(self.model_names, self.schedulers)):
            old_lr = self.optimizers[i].param_groups[0]['lr']
            if self.train_config["GAN"]["lr_policy"] == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
            lr = self.optimizers[i].param_groups[0]['lr']
            print(f'{name}: learning rate {old_lr:.7f} -> {lr:.7f}')

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

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

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
            parser.add_argument('--lambda_sign', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_audio', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')

        return parser

    def __init__(self, args, preprocess_config, model_config, train_config, configs_ft=None, isTrain=True):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, args, preprocess_config, model_config, train_config, isTrain)

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_sign2audio = Sign2Speech(preprocess_config, model_config)
        self.netG_sign2audio = Sign2Speech(preprocess_config, model_config)
        self.model_names.append("G_sign2audio")
        networks.init_net(self.netG_sign2audio, gpu_ids=self.gpu_ids)
        if configs_ft is not None:
            train_config_ft = configs_ft[2]
            ckpt_path = os.path.join(
                train_config_ft["path"]["ckpt_path"],
                "{}.pth.tar".format(args.restore_step_ft),
            )
            print(f"Load {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            self.netG_sign2audio.load_state_dict(ckpt["model"], strict=False)

        self.audio2sign = None

        self.step = 0

        if self.isTrain:  # define discriminators
            self.netD_audio = networks.define_D(
                model_config["D_audio"]["input_nc"], model_config["D_audio"]["ndf"], "audio",
                model_config["D_audio"]["n_layers_D"], model_config["D_audio"]["norm"],
                model_config["D_audio"]["init_type"], model_config["D_audio"]["init_gain"], train_config["gpu_ids"])
            self.model_names.append("D_audio")

        if self.isTrain:
            self.fake_audio_pool = AudioPool(train_config["GAN"]["pool_size"])  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(train_config["GAN"]["gan_mode"]).to(self.device)  # define GAN loss.
            # self.criterionCycle = torch.nn.L1Loss()
            # self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            train_parameters = self.netG_sign2audio.parameters()
            # train_layers = ["sign_processer", "s2s_mixier", "visual_project"]
            # for name, param in self.netG_sign2audio.named_parameters():
            #     if any(layer_name in name for layer_name in train_layers):
            #         train_parameters.append(param)

            self.optimizer_G = torch.optim.Adam(train_parameters,
                                                lr=train_config["optimizer"]["lr_G_s2a"],
                                                betas=train_config["optimizer"]["betas"])
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_sign2audio.parameters(),
            #                                                     self.audio2sign.parameters()),
            #                                     lr=train_config["optimizer"]["lr"],
            #                                     betas=train_config["optimizer"]["betas"])
            self.optimizer_D = torch.optim.Adam(self.netD_audio.parameters(),
                                                lr=train_config["optimizer"]["lr_D_a"],
                                                betas=train_config["optimizer"]["betas"])
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, inputs):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            inputs (dict): include the data itself and its metadata information.
        """
        self.real_sign  = inputs["sign"]
        self.fake_raw_texts = self.real_sign[0]
        self.fake_text_lens = self.real_sign[4]

        self.real_audio = inputs["audio"]
        self.real_ids = self.real_audio[0]
        self.real_raw_texts = self.real_audio[1]
        self.real_speakers = self.real_audio[2]
        self.real_texts = self.real_audio[3]
        self.real_text_lens = self.real_audio[4]
        self.real_max_text_lens = torch.Tensor(self.real_audio[5])
        self.real_mels = self.real_audio[6]
        self.real_audio_lens = self.real_audio[7]
        self.real_max_mel_lens = torch.Tensor(self.real_audio[8])
        self.real_pitches = self.real_audio[9]
        self.real_energies = self.real_audio[10]
        self.real_durations = self.real_audio[11]

        # self.set_eval_mode()
        # self.real_speaker_text_embedding = self.netG_sign2audio.text_encoding(
        #     self.real_speakers.to(self.device),
        #     self.real_texts.to(self.device),
        #     self.real_text_lens.to(self.device),
        #     self.real_max_text_lens.to(self.device),
        # )

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length = self.real_sign
        batch_size = text_tokens.shape[0]
        max_src_len = token_length.max()
        text_tokens = text_tokens[:, :max_src_len]

        speakers = torch.randint(self.speaker_num, (batch_size,)).long()
        (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            speaker_text_embedding,
        ) = self.netG_sign2audio(
            speakers.to(self.device),
            text_tokens.to(self.device),
            token_length.to(self.device),
            max_src_len,
            key_point=visual_prefix.to(self.device)
        )  # G_A(A)

        # self.fake_speaker_text_embedding = speaker_text_embedding
        self.fake_audio = postnet_output.masked_fill(
            mel_masks.unsqueeze(2).repeat(1, 1, postnet_output.shape[2]), 0.0).unsqueeze(1)
        self.fake_audio_lens = mel_lens
        # self.rec_sign = self.audio2sign(self.fake_audio)   # G_B(G_A(A))

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
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_audio(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_audio = self.fake_audio_pool.query(self.fake_audio)
        real_audio = self.real_mels.to(self.device)
        self.loss_D_audio = self.backward_D_basic(
            self.netD_audio,
            real_audio.unsqueeze(1),
            fake_audio.unsqueeze(1)
            )

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        # GAN loss D_audio(G_sign2audio(sign))
        pred_fake = self.netD_audio(self.fake_audio)
        self.loss_G_audio = self.criterionGAN(pred_fake, True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_sign, self.real_sign) * lambda_sign
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_audio  # + self.loss_cycle_A
        self.loss_G.backward()

    def calc_confusion_matrix(self):
        self.set_eval_mode()
        real_audio = self.real_mels.to(self.device)
        pred_real = self.netD_audio(real_audio.unsqueeze(1))
        pred_fake = self.netD_audio(self.fake_audio)
        pred_real = pred_real >= 0.5
        pred_fake = pred_fake >= 0.5
        gt_real = torch.full_like(pred_real, True)
        gt_fake = torch.full_like(pred_fake, False)
        preds = torch.concat((pred_real, pred_fake))
        gt = torch.concat((gt_real, gt_fake))
        return preds.squeeze(1).detach().cpu().numpy(), gt.squeeze(1).detach().cpu().numpy()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.set_train_mode()
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_audio], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        if self.step == 0:
            # D_A and D_B
            self.set_requires_grad([self.netD_audio], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_audio()      # calculate gradients for D_audio
            self.optimizer_D.step()  # update D_A and D_B's weights
        self.step = (self.step + 1) % 2
