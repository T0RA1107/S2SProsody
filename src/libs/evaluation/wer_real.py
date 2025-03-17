import os
import glob
import tqdm

from transformers import pipeline
from evaluate import load


def main():
    pipe = pipeline("automatic-speech-recognition", "openai/whisper-large-v3", device="cuda:0")
    wer = load("wer")
    target = ["p340", "p360"]
    data_dir = "/home/aolab/Desktop/S2SProsody/data/VCTK-Corpus/raw"
    predictions = []
    references = []
    for speaker in target:
        wav_path_list = glob.glob(os.path.join(data_dir, "wav48", speaker, "*.wav"))
        txt_path_list = glob.glob(os.path.join(data_dir, "txt", speaker, "*.txt"))
        for wav, txt in tqdm.tqdm(zip(wav_path_list, txt_path_list), total=len(wav_path_list)):
            txt = open(txt, "r").readline().rstrip("\n")
            pred = pipe(wav)["text"]
            predictions.append(pred)
            references.append(txt)
    wer_score = wer.compute(predictions=predictions, references=references)
    print(wer_score)


if __name__ == "__main__":
    main()
