# S2SProsody

Sign language video to speech synthesis with prosody modeling. Given sign language pose sequences, the model generates natural-sounding speech that reflects the prosody (pitch and energy patterns) implied by the signing.

## Overview

S2SProsody extends [FastSpeech2](https://github.com/ming024/FastSpeech2) with a sign-aware prosody control mechanism:

1. **Pose Backbone** — extracts visual features from sign language keypoints (CT-GCN via [GloFE](https://github.com/HenryLittle/GloFE))
2. **S2SMixer** — cross-attention transformer that injects sign features into the speech encoder
3. **MoE (Mixture of Experts)** — routes sign features to control pitch/energy prediction
4. **FastSpeech2** — non-autoregressive TTS backbone for mel-spectrogram generation
5. **HiFi-GAN** — neural vocoder for mel-to-waveform conversion
6. **Audio Discriminator** — adversarial training for speaker likeness

The model is trained on paired (sign video, speech audio) data from [OpenASL](https://github.com/chevalierNoir/OpenASL) with speaker reference from [VCTK](https://datashare.ed.ac.uk/handle/10283/3443).

## Requirements

- Python 3.9+
- PyTorch 1.13.1
- CUDA 11.8
- GPU with at least 16GB VRAM (3× GPU recommended for training)

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd S2SProsody

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### 1. VCTK Corpus (speech reference)

Download from [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) and place under `data/VCTK-Corpus/`.

```
data/VCTK-Corpus/
├── wav48_silence_trimmed/   # raw audio
└── txt/                     # transcriptions
```

Run MFA alignment and preprocessing:

```bash
cd src
python preprocess.py -p config/VCTK/preprocess.yaml
```

### 2. OpenASL (sign language)

Download from [OpenASL](https://github.com/chevalierNoir/OpenASL) and place under `data/OpenASL/`.

```
data/OpenASL/
└── data/
    ├── openasl-v1.0.tsv      # metadata
    ├── translation_token_ids.txt
    ├── video-clip/            # sign language video clips
    └── mmpose/                # extracted pose keypoints
```

Extract pose keypoints with MMPose:

```bash
bash src/libs/preprocessing/mmpose/extract.sh
```

### 3. HiFi-GAN Pretrained Vocoder

Download pretrained HiFi-GAN weights:

```bash
bash src/scripts/download_hifigan.sh
```

This places model weights under `src/libs/models/hifigan/`.

## Training

### Step 1: Pretrain on VCTK (TTS only)

```bash
cd src
bash scripts/pretrain.sh
```

### Step 2: Fine-tune with sign language input

```bash
cd src
bash scripts/train.sh
```

Training configuration is in `src/config/Sign2Speech/train.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `optimizer.batch_size` | 10 | Batch size per GPU |
| `step.total_step` | 900000 | Total training steps |
| `loss.weight.intonation` | 10 | Intonation loss weight |
| `loss.weight.speaker_likeness` | 10 | Speaker likeness loss weight |
| `GAN.gan_mode` | lsgan | GAN objective type |

Training logs are tracked with [Weights & Biases](https://wandb.ai/).

## Inference

```bash
cd src
python inference.py \
  -p config/Sign2Speech/preprocess.yaml \
  -m config/Sign2Speech/model.yaml \
  -t config/Sign2Speech/train.yaml \
  --ckpt_path <path/to/checkpoint.pth> \
  --output_dir <path/to/output>
```

## Evaluation

Evaluate prosody metrics (pitch/energy DTW):

```bash
cd src
bash scripts/evaluate_prosody.sh
```

Evaluate word error rate (WER) using Whisper:

```bash
cd src
python libs/evaluation/wer_real.py --data_dir <path/to/vctk/raw>
```

## Project Structure

```
src/
├── train.py                        # Training entry point
├── inference.py                    # Inference/evaluation
├── config/                         # YAML configuration files
│   ├── Sign2Speech/                # Main S2S task configs
│   ├── VCTK/                       # VCTK pretraining configs
│   └── LibriTTS/                   # LibriTTS configs
├── libs/
│   ├── models/
│   │   ├── sign2speech.py          # Main model
│   │   ├── fastspeech2.py          # TTS backbone
│   │   ├── semi_cycle_gan.py       # GAN training wrapper
│   │   ├── visual_backbone/        # CT-GCN pose encoder
│   │   ├── modules/                # Transformer, variance adaptor
│   │   ├── hifigan/                # Vocoder
│   │   └── loss_fn/                # Loss functions
│   ├── util/                       # Utilities
│   ├── evaluation/                 # Evaluation metrics
│   └── preprocessing/              # Data preprocessing
└── dataset/                        # Dataset classes
```

## Dependencies

- [GloFE](https://github.com/HenryLittle/GloFE) — sign language feature extraction
- [FastSpeech2](https://github.com/ming024/FastSpeech2) — TTS backbone
- [PyTorch Template](https://github.com/yiskw713/pytorch_template) — training framework

## Citation

```bibtex
@misc{s2sprosody,
  title  = {S2SProsody: Sign-to-Speech Synthesis with Prosody Modeling},
  author = {},
  year   = {2024},
}
```
