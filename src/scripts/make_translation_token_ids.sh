export OpenASL_PATH=$1
export VCTK_PATH=$2

echo $OpenASL_PATH
echo $VCTK_PATH

python make_translation_token_ids.py\
    --output_path "${OpenASL_PATH}/data/translation_token_ids.txt"\
    --label_path "${OpenASL_PATH}/data/openasl-v1.0.tsv"\
    --lexicon_path "${VCTK_PATH}/processed/mfa_dict.txt"\
    --text_cleaners english_cleaners
