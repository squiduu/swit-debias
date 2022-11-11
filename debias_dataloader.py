import torch
from typing import List, Union
from logging import Logger
from torch.utils.data import DataLoader, Dataset
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from debias_config import DataArguments, ModelArguments, TrainingArguments
from debias_utils import load_sentences, clear_sentences, load_words, clear_words, tokenize_prompts, to_cuda


class DebiasDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data: List[dict]):
        """Reads source and target sequences from txt files."""
        self.data = data

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]

        return item_info

    def __len__(self):
        return len(self.data)


def read_data(
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer], data_args: DataArguments, logger: Logger
) -> List[dict]:
    if "wikipedia" in data_args.prompt_dir:
        logger.info("Load prompts from wikipedia.")
        _prompts = load_sentences(path=data_args.prompt_dir)
        prompts = clear_sentences(_sentence=_prompts, tokenizer=tokenizer)
    else:
        logger.info("Load searched biased prompts.")
        prompts = load_words(path=data_args.prompt_dir)

    logger.info(f"Load attribute words for {data_args.debias_type}.")
    if data_args.debias_type == "gender":
        _targ1_words = load_words(path=data_args.data_dir + "male.txt")
        _targ2_words = load_words(path=data_args.data_dir + "female.txt")

    elif data_args.debias_type == "race":
        _targ1_words = load_words(path=data_args.data_dir + "af_american.txt")
        _targ2_words = load_words(path=data_args.data_dir + "eu_american.txt")

    targ1_words, targ2_words = clear_words(_words1=_targ1_words, _words2=_targ2_words, tokenizer=tokenizer)

    logger.info("Get inputs for fine-tuning.")
    targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx = tokenize_prompts(
        prompts=prompts, targ1_words=targ1_words, targ2_words=targ2_words, tokenizer=tokenizer
    )

    logger.info("Send all input tensors to cuda.")
    targ1_tokens, targ2_tokens = to_cuda(targ1_tokens=targ1_tokens, targ2_tokens=targ2_tokens)

    logger.info("Get a list of total model inputs.")
    data = []
    for batch_idx in range(torch.Tensor.size(targ1_tokens["input_ids"])[0]):
        data.append(
            {
                "targ1_input_ids": targ1_tokens["input_ids"][batch_idx],
                "targ1_attention_mask": targ1_tokens["attention_mask"][batch_idx],
                "targ1_token_type_ids": targ1_tokens["token_type_ids"][batch_idx],
                "targ1_mask_idx": targ1_mask_idx[batch_idx],
                "targ2_input_ids": targ2_tokens["input_ids"][batch_idx],
                "targ2_attention_mask": targ2_tokens["attention_mask"][batch_idx],
                "targ2_token_type_ids": targ2_tokens["token_type_ids"][batch_idx],
                "targ2_mask_idx": targ2_mask_idx[batch_idx],
            }
        )

    return data


def get_dataloaders(
    data: List[dict],
    data_args: DataArguments,
    model_args: ModelArguments,
    train_args: TrainingArguments,
    logger: Logger,
) -> DataLoader:
    logger.info("Make a dataset.")
    dataset = DebiasDataset(data=data, data_args=data_args, model_args=model_args, train_args=train_args)

    logger.info("Make a dataloader.")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=train_args.per_device_batch_size,
        shuffle=True,
        num_workers=train_args.dataloader_num_workers,
        pin_memory=True,
    )

    return dataloader
