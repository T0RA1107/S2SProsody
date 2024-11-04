import argparse
import os
import datetime

import soundfile as sf
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from FastSpeech2.evaluate import evaluate
from FastSpeech2.utils.model import get_vocoder, vocoder_infer

from models.semi_cycle_gan import SemiCycleGANModel
from dataset import UnpairedAudioSignDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs, configs_ft):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    dataset = UnpairedAudioSignDataset(
        "train.txt", preprocess_config, train_config, args)  # create a dataset given opt.dataset_mode and other options
    if "speaker_num" not in model_config:
        model_config["speaker_num"] = dataset.speaker_num
    dataset_size = len(dataset)    # get the number of images in the dataset.
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    print('The number of training images = %d' % dataset_size)

    model = SemiCycleGANModel(args, preprocess_config, model_config, train_config, configs_ft)      # create a model given opt.model and other options
    model.setup(train_config)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    vocoder = get_vocoder(model_config, device)

    dt_now = datetime.datetime.now()
    run_name = dt_now.strftime('%m:%d:%H:%M')
    run_dir = f"./output/{run_name}/"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir + "wav_preds/", exist_ok=True)
    pred_txt_file = run_dir + "wav_preds/pred.txt"

    if args.use_wandb:
        wandb.init(
            project='Sign2Speech',
            group="length test",
            job_type="training",
            name=run_name,
            config={
                "preprocess": preprocess_config,
                "model": model_config,
                "train": train_config
            },
        )
        table = wandb.Table(columns=["translation", "Speech"])

    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    n_epochs_decay = train_config["step"]["n_epochs_decay"]
    step_count = train_config["step"]["step_count"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    assert synth_step % log_step == 0
    assert val_step % log_step == 0

    for epoch in tqdm(range(step_count, total_step + n_epochs_decay + 1), desc="Training"):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for batchs in tqdm(loader, leave=False, desc=f"EPOCH {epoch}"):  # inner loop within one epoch
            for batch in batchs:

                total_iters += len(batch)
                epoch_iter += len(batch)
                model.set_input(batch)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights


                if total_iters % log_step == 0:    # print training losses and save logging information to the disk
                    log = {}

                    loss_G = model.loss_G_audio
                    loss_D = model.loss_D_audio

                    log["loss/G_audio"] = loss_G
                    log["loss/D_audio"] = loss_D
                    log["loss/total_reconstruction"] = model.total_loss_reconstruction
                    log["loss/mel_reconstruction"] = model.mel_loss_reconstruction
                    log["loss/postnet_mel_reconstruction"] = model.postnet_mel_loss_reconstruction
                    log["loss/pitch_reconstruction"] = model.pitch_loss_reconstruction
                    log["loss/energy_reconstruction"] = model.energy_loss_reconstruction
                    log["loss/duration_reconstruction"] = model.duration_loss_reconstruction

                    # calc confusion matrix
                    preds, gt = model.calc_confusion_matrix()
                    log["confusion_matrix"] = wandb.plot.confusion_matrix(probs=None,
                        y_true=gt, preds=preds, class_names=["False", "True"])

                    # log the distribution of predict audio length
                    length = model.fake_audio_with_sign_lens.float()
                    mean = torch.mean(length)
                    var = torch.mean((length - mean) ** 2.)
                    log["length/mean_text"] = model.fake_text_lens.float().mean()
                    log["length/mean_pred"] = mean
                    log["length/var_pred"] = var
                    # print("predict length")
                    # print(mean, var)

                    # log the distribution of GT audio length
                    length = model.real_audio_lens.float()
                    mean = torch.mean(length)
                    var = torch.mean((length - mean) ** 2.)
                    log["length/mean_text_GT"] = model.real_text_lens.float().mean()
                    log["length/mean_GT"] = mean
                    log["length/var_GT"] = var
                    # print("predict length")
                    # print(mean, var)

                if total_iters % synth_step == 0:
                    output = model.fake_audio_with_sign
                    output_lens = model.fake_audio_with_sign_lens
                    raw_text = model.fake_raw_texts[0]
                    mel_len = output_lens[0].item()
                    mel_prediction = output[0, :, :mel_len].detach().transpose(1, 2)
                    wav_prediction = vocoder_infer(
                        mel_prediction,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )[0]
                    # audio = wandb.Audio(
                    #     wav_prediction / max(abs(wav_prediction)),
                    #     sample_rate=sampling_rate)
                    sf.write(
                        run_dir + f"wav_preds/synth_{total_iters // synth_step}.wav",
                        wav_prediction, samplerate=sampling_rate)
                    with open(pred_txt_file, "a") as f:
                        f.write(raw_text + "\n")

                    # table.add_data(raw_text, audio)
                    # log["Training"] = table

                if args.use_wandb and total_iters % log_step == 0:
                    wandb.log(log)

                    # visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                # if total_iters % save_step == 0:   # cache our latest model every <save_latest_freq> iterations
                #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                #     save_suffix = 'iter_%d' % total_iters
                #     model.save_networks(save_suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--isTrain", action="store_true"
    )
    parser.add_argument(
        "--use_wandb", action="store_true"
    )
    parser.add_argument(
        "--fine_tuning", action="store_true"
    )
    parser.add_argument(
        "--restore_step_ft", type=int
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    configs_ft = None
    if args.fine_tuning:
        ft_config_path = train_config["fine-tuning"]
        preprocess_config_ft = yaml.load(open(
            os.path.join(ft_config_path, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
        model_config_ft = yaml.load(open(
            os.path.join(ft_config_path, "model.yaml"), "r"), Loader=yaml.FullLoader)
        train_config_ft = yaml.load(open(
            os.path.join(ft_config_path, "train.yaml"), "r"), Loader=yaml.FullLoader)
        configs_ft = (preprocess_config_ft, model_config_ft, train_config_ft)

    main(args, configs, configs_ft)
