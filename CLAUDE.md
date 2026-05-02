# CLAUDE.md

## Development Environment

Always activate `.venv` before running Python:

```bash
source .venv/bin/activate
```

## Architecture

### Model Pipeline

```
Sign video keypoints
    → PartedPoseBackbone (CT-GCN)       # visual features
    → S2SMixer (cross-attention)         # injects sign into speech encoder
    → MoE (Mixture of Experts)           # routes sign features
    → VarianceAdaptorWithReference       # pitch/energy prediction with sign reference
    → FastSpeech2 decoder                # mel-spectrogram generation
    → HiFi-GAN vocoder                   # waveform synthesis
```

### Key Classes

| Class | File | Role |
|---|---|---|
| `Sign2Speech` | `src/libs/models/sign2speech.py` | Top-level model |
| `FastSpeech2` | `src/libs/models/fastspeech2.py` | TTS backbone |
| `SemiCycleGAN` | `src/libs/models/semi_cycle_gan.py` | GAN training logic |
| `PartedPoseBackbone` | `src/libs/models/visual_backbone/` | CT-GCN pose encoder |
| `S2SMixer` | `src/libs/models/modules/transformer/Models.py` | Sign-speech cross-attention |
| `MoE` | `src/libs/models/modules/transformer/Models.py` | Mixture of Experts routing |

### Configuration

Three YAML files per dataset under `src/config/<dataset>/`:
- `preprocess.yaml` — data paths and audio/text preprocessing parameters
- `model.yaml` — network architecture hyperparameters
- `train.yaml` — optimizer, loss weights, training schedule

### Loss Components

- `prosody` — distribution matching loss (pitch/energy)
- `intonation` — contrastive loss between sign-conditioned and unconditioned outputs
- `speaker_likeness` — GAN-based speaker consistency
- `gate` — MoE gate regularization

## Running Tests

```bash
source .venv/bin/activate
pytest -v --cov=src --cov-report term-missing
```

## Linting and Type Checking

```bash
source .venv/bin/activate
mypy . --ignore-missing-imports
flake8 src/
black src/
isort src/
```

## Pre-commit

```bash
pre-commit run --all-files
```

## Training Data Layout

```
data/
├── VCTK-Corpus/
│   ├── wav48_silence_trimmed/   # raw audio
│   ├── txt/                     # transcriptions
│   └── preprocess/              # preprocessed binary (generated)
└── OpenASL/
    └── data/
        ├── openasl-v1.0.tsv
        ├── translation_token_ids.txt
        ├── video-clip/
        └── mmpose/              # extracted keypoints
```

## Outputs

- Checkpoints: `output/<dataset>/`
- Inference results: `result/<dataset>/`
- W&B logs: `src/wandb/` (not committed)
