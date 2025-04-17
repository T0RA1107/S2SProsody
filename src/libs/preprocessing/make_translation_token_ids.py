import argparse
import re
from string import punctuation
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
from g2p_en import G2p

from libs.preprocessing.speech.text import text_to_sequence


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, args):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(args.lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    return text_to_sequence(
            phones, args.text_cleaners
        )


def write_txt(pack):
    i, (vid, trans, args) = pack
    trans_ids = preprocess_english(trans, args)
    trans_ids_txt = "{}|{}\n".format(vid, " ".join(map(str, trans_ids)))
    return i, trans_ids_txt


def main(args):
    data_frame = pd.read_csv(args.label_path, sep="\t", low_memory=False)
    translation = data_frame["raw-text"].to_list()
    video_names = data_frame["vid"].to_list()

    pool = Pool(processes=32)
    trans_ids_txt_list = ["" for _ in range(len(translation))]
    pbar = tqdm(total=len(translation), desc="Make translation ids", leave=False)

    for i, trans_ids_txt in pool.imap_unordered(write_txt, enumerate(zip(video_names, translation, [args] * len(translation)))):
        trans_ids_txt_list[i] = trans_ids_txt
        pbar.update(1)
    with open(args.output_path, "w") as f:
        f.writelines(trans_ids_txt_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, required=True
    )
    parser.add_argument(
        "--label_path", type=str, required=True
    )
    parser.add_argument(
        "--lexicon_path", type=str, required=True
    )
    parser.add_argument(
        "--text_cleaners", type=str, nargs="*", required=True
    )
    args = parser.parse_args()
    main(args)
