#!/usr/bin/env bash
# Download pretrained HiFi-GAN vocoder weights from the official repository.
# Source: https://github.com/jik876/hifi-gan

set -e

HIFIGAN_DIR="$(cd "$(dirname "$0")/../libs/models/hifigan" && pwd)"

echo "Downloading HiFi-GAN pretrained weights to $HIFIGAN_DIR ..."

# Universal model (recommended for multi-speaker synthesis)
wget -q --show-progress \
  "https://github.com/jik876/hifi-gan/releases/download/v1/g_02500000" \
  -O "$HIFIGAN_DIR/generator_universal.pth.tar"

# LJSpeech model (single-speaker)
# Uncomment if needed:
# wget -q --show-progress \
#   "https://github.com/jik876/hifi-gan/releases/download/v1/g_02500000" \
#   -O "$HIFIGAN_DIR/generator_LJSpeech.pth.tar"

echo "Done. Weights saved to $HIFIGAN_DIR"
