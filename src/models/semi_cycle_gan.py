import torch
import torch.nn as nn
import os
from collections import OrderedDict
import itertools
import numpy as np

from util.audio_pool import AudioPool
from . import networks
from .sign2audio import Sign2Speech
from .prosody_estimator import ProsodyEstimator1D, ProsodyEstimator2D

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

    def get_learning_rate(self):
        lr_dict = {}
        for i, name  in enumerate(self.model_names):
            lr = self.optimizers[i].param_groups[0]['lr']
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

            self.netProsody_estimator = ProsodyEstimator1D(
                model_config["D_audio"]["input_nc"], 4
            )
            networks.init_net(self.netProsody_estimator, gpu_ids=self.gpu_ids)

            self.weight_reconstruction = train_config["loss"]["weight"]["reconstruction"]
            self.mean = train_config["discriminator"]["noise"]["mean"]
            self.std = train_config["discriminator"]["noise"]["std"]
            self.fake_audio_pool = AudioPool(train_config["GAN"]["pool_size"])  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(train_config["GAN"]["gan_mode"]).to(self.device, non_blocking=True)  # define GAN loss.
            self.criterionProsody = nn.MSELoss(reduction='none').to(self.device, non_blocking=True)
            self.fastspeech2loss = FastSpeech2Loss(preprocess_config, model_config)
            # self.criterionCycle = torch.nn.L1Loss()
            # self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            train_parameters = list(self.netG_sign2audio.parameters())
            train_parameters += list(self.netProsody_estimator.parameters())
            # train_parameters = []
            # train_layers = ["sign_processer", "s2s_mixier", "visual_project"]
            # for name, param in self.netG_sign2audio.named_parameters():
            #     if any(layer_name in name for layer_name in train_layers):
            #         train_parameters.append(param)
            # train_parameters += list(self.netProsody_estimator.parameters())

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
        self.prosody_label = self.real_sign[6].to(self.device, non_blocking=True)

        ###  Real Audio  ###
        # self.real_audio = inputs["audio"]
        # self.real_ids = self.real_audio[0]
        # self.real_raw_texts = self.real_audio[1]
        # self.real_speakers = self.real_audio[2].long().to(self.device, non_blocking=True)
        # self.real_texts = self.real_audio[3].long().to(self.device, non_blocking=True)
        # self.real_text_lens = self.real_audio[4].long().to(self.device, non_blocking=True)
        # self.real_max_text_lens = torch.Tensor(self.real_audio[5]).long().to(self.device, non_blocking=True)
        # self.real_mels = self.real_audio[6].float().to(self.device, non_blocking=True)
        # self.real_audio_lens = self.real_audio[7].long().to(self.device, non_blocking=True)
        # self.real_max_mel_lens = torch.Tensor(self.real_audio[8]).long().to(self.device, non_blocking=True)
        # self.real_pitches = self.real_audio[9].float().to(self.device, non_blocking=True)
        # self.real_energies = self.real_audio[10].float().to(self.device, non_blocking=True)
        # self.real_durations = self.real_audio[11].long().to(self.device, non_blocking=True)
        # self.real_mel_masks = make_mask_from_lens(self.real_audio_lens, self.real_max_mel_lens)

        # self.real_audio = (
        #     self.real_ids,
        #     self.real_raw_texts,
        #     self.real_speakers,
        #     self.real_texts,
        #     self.real_text_lens,
        #     self.real_max_text_lens,
        #     self.real_mels,
        #     self.real_audio_lens,
        #     self.real_max_mel_lens,
        #     self.real_pitches,
        #     self.real_energies,
        #     self.real_durations
        # )
        ###  Real Audio  ###

        # self.set_eval_mode()
        # self.real_speaker_text_embedding = self.netG_sign2audio.text_encoding(
        #     self.real_speakers.to(self.device, non_blocking=True),
        #     self.real_texts.to(self.device, non_blocking=True),
        #     self.real_text_lens.to(self.device, non_blocking=True),
        #     self.real_max_text_lens.to(self.device, non_blocking=True),
        # )

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, prosody_label = self.real_sign
        batch_size = text_tokens.shape[0]
        token_length = token_length.to(self.device, non_blocking=True)
        max_src_len = token_length.max().to(self.device, non_blocking=True)
        text_tokens = text_tokens[:, :max_src_len].to(self.device, non_blocking=True)

        speakers = torch.randint(self.speaker_num, (batch_size,), device=self.device).long()

        # without sign language TTS
        with torch.no_grad():
            (
                _,
                postnet_output,
                _,
                _,
                _,
                _,
                _,
                mel_masks,
                _,
                mel_lens,
            ) = self.netG_sign2audio(
                speakers,
                text_tokens,
                token_length,
                max_src_len,
                key_point=None
            )

            self.synth_mel_masks = make_mask_from_lens(mel_lens, max_length=postnet_output.shape[1])
            self.synth_audio = postnet_output.masked_fill(
                mel_masks.unsqueeze(2).repeat(1, 1, postnet_output.shape[2]), 0.0).unsqueeze(1)

            self.synth_audio_lens = mel_lens.detach().cpu()

        (
            _,
            postnet_output,
            _,  # p_predictions,
            e_predictions,
            _,
            _,
            _,
            mel_masks,
            _,
            mel_lens,
        ) = self.netG_sign2audio(
            speakers,
            text_tokens,
            token_length,
            max_src_len,
            key_point=visual_prefix.to(self.device, non_blocking=True)
        )  # G_A(A)

        # self.fake_speaker_text_embedding = speaker_text_embedding
        self.fake_mel_masks = make_mask_from_lens(mel_lens, max_length=postnet_output.shape[1])
        self.fake_audio_with_sign = postnet_output.masked_fill(
            mel_masks.unsqueeze(2).repeat(1, 1, postnet_output.shape[2]), 0.0).unsqueeze(1)

        self.fake_audio_with_sign_lens = mel_lens
        self.speakers = speakers.detach().cpu()
        # n_bins = self.netG_sign2audio.variance_adaptor.energy_bins.shape[0]
        # e_bucket = torch.bucketize(e_predictions, self.netG_sign2audio.variance_adaptor.energy_bins)
        # self.e_histogram = torch.histc(e_bucket, bins=n_bins, min=0, max=n_bins-1)
        # self.e_histogram /= self.e_histogram.sum() + 1e-5
        self.pred_prosody_label = self.netProsody_estimator(e_predictions.unsqueeze(1))

        # synthesize audio for reconstruction
        # self.fake_audio_GT = self.netG_sign2audio(
        #     self.real_speakers,
        #     self.real_texts,
        #     self.real_text_lens,
        #     self.real_max_text_lens,
        #     mel_lens=self.real_audio_lens,
        #     max_mel_len=self.real_max_mel_lens,
        #     p_targets=self.real_pitches,
        #     e_targets=self.real_energies,
        #     d_targets=self.real_durations,
        #     key_point=None
        # )

        # self.rec_sign = self.audio2sign(self.fake_audio)   # G_B(G_A(A))
        del speakers, text_tokens, token_length, max_src_len
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
        self.loss_D_audio = loss_D.detach().cpu()
        del pred_real, loss_D_real, pred_fake, loss_D_fake, loss_D
        torch.cuda.empty_cache()

    def backward_D_audio(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_audio = self.fake_audio_pool.query(
            self.fake_audio_with_sign.detach().cpu()).to(
                self.device, non_blocking=True)
        real_audio = self.synth_audio
        self.backward_D_basic(
            self.netD_audio,
            real_audio,
            fake_audio.unsqueeze(1)
        )
        del fake_audio, real_audio
        torch.cuda.empty_cache()

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        # GAN loss D_audio(G_sign2audio(sign))
        # fake_audio = self.fake_audio_with_sign
        # pred_fake = self.netD_audio(self.augmentation_audio(fake_audio, self.fake_mel_masks))
        # loss_G_audio = self.criterionGAN(pred_fake, True)
        # self.loss_G_audio = loss_G_audio.detach().cpu()
        # reconstruct GT audio
        # loss_G_reconstruction = self.fastspeech2loss(self.real_audio, self.fake_audio_GT)
        # self.total_loss_reconstruction = loss_G_reconstruction[0].detach().cpu()
        # self.mel_loss_reconstruction = loss_G_reconstruction[1].detach().cpu()
        # self.postnet_mel_loss_reconstruction = loss_G_reconstruction[2].detach().cpu()
        # self.pitch_loss_reconstruction = loss_G_reconstruction[3].detach().cpu()
        # self.energy_loss_reconstruction = loss_G_reconstruction[4].detach().cpu()
        # self.duration_loss_reconstruction = loss_G_reconstruction[5].detach().cpu()

        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_prosody = self.criterionProsody(self.pred_prosody_label, self.prosody_label)
        self.v_max_loss = loss_prosody[:, :2].mean().detach().cpu()
        self.a_max_loss = loss_prosody[:, 2:4].mean().detach().cpu()
        loss_prosody = loss_prosody.mean()
        # combined loss and calculate gradients
        # loss_G = loss_G_audio
        loss_G = 0.
        # loss_G += loss_G_reconstruction[0] * self.weight_reconstruction
        loss_G += loss_prosody

        self.loss_prosody = loss_prosody.detach().cpu()

        # self.loss_G = loss_G.detach().cpu()

        loss_G.backward()
        # del fake_audio, pred_fake, loss_G_audio,
        del loss_prosody, loss_G
        torch.cuda.empty_cache()

    def calc_confusion_matrix(self):
        self.set_eval_mode()
        real_audio = self.real_mels.to(self.device, non_blocking=True)
        pred_real = self.netD_audio(self.augmentation_audio(real_audio.unsqueeze(1), self.real_mel_masks))
        pred_fake = self.netD_audio(self.augmentation_audio(self.fake_audio_with_sign, self.fake_mel_masks))
        pred_real = (pred_real >= 0.5).sum().detach().cpu()
        pred_fake = (pred_fake >= 0.5).sum().detach().cpu()

        bs = real_audio.shape[0]
        cm = np.array([
            [pred_fake, bs - pred_fake],
            [bs - pred_real, pred_real]
        ])
        return cm

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.set_train_mode()
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_audio], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad(set_to_none=True)  # set G's gradients to zero
        self.backward_G()             # calculate gradients for G
        self.optimizer_G.step()       # update G's weights
        # if self.step == 0:
        #     # D_A and D_B
        #     self.set_requires_grad([self.netD_audio], True)
        #     self.optimizer_D.zero_grad(set_to_none=True)   # set D's gradients to zero
        #     self.backward_D_audio()      # calculate gradients for D_audio
        #     self.optimizer_D.step()  # update D_A and D_B's weights
        # self.step = (self.step + 1) % 2
        torch.cuda.empty_cache()
