# swit-debias
Scripts of "Switching [MASK] Tokens for Gender Debiasing in Pre-trained Language Models."

## Installation
This repository is available in Ubuntu 20.04 LTS, and it is not tested in other OS.
```
git clone https://github.com/squiduu/swit-debias.git
cd swit-debias

conda create -n switdebias python=3.7.10
conda activate switdebias

pip install -r requirements.txt
```

## Bias mitigation
### Biased prompt searching
Search biased prompts with pre-defined sets of words.
```
mkdir ./out/
sh prompts_for_finetune.sh
```
Then you will get a prompt file at `./data/prompts_bert-base-uncased_gender.txt`

### Debiaing
Fine-tune a pre-trained BERT to debias.
```
sh debias.sh
```
Then you will get two checkpoint files at `./out/glue_bert-base-uncased_run00_gender/` and `./out/seat_bert-base-uncased_run00_gender/`
