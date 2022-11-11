import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import apex.amp as amp
import numpy as np
from tqdm import tqdm
from logging import Logger
from typing import Union, Tuple, List, Dict
from torch.nn.parallel.data_parallel import DataParallel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.albert.modeling_albert import AlbertModel
from transformers.tokenization_utils_base import BatchEncoding
from prompts_config import DataArguments, ModelArguments, TrainingArguments
from modelings import BertForDebias, RobertaForDebias, AlbertForDebias


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"
    os.system(command)


def get_logger(train_args: TrainingArguments) -> Logger:
    """Create and set environments for logging.

    Args:
        train_args (TrainingArguments): A parsed arguments.

    Returns:
        logger (Logger): A logger for checking progress.
    """
    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmtr = logging.Formatter(fmt="%(asctime)s | %(module)s | %(levelname)s > %(message)s", datefmt="%Y-%m-%d %H:%M")
    # handler for console
    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(fmtr)
    logger.addHandler(console_hdlr)
    # handler for .log file
    os.makedirs(train_args.output_dir, exist_ok=True)
    file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"prompt_{train_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run name: {train_args.run_name}")

    return logger


def prepare_model_and_tokenizer(
    model_name_or_path: str, data_args: DataArguments, model_args: ModelArguments
) -> Tuple[
    Union[BertForDebias, RobertaForDebias, AlbertForDebias],
    Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
]:
    """Download and prepare the pre-trained model and tokenizer.

    Args:
        model_name_or_path (str): A version of pre-trained model.
    """
    if model_args.model_name == "bert":
        model_class = BertForDebias
        tokenizer_class = BertTokenizer
    elif model_args.model_name == "roberta":
        model_class = RobertaForDebias
        tokenizer_class = RobertaTokenizer
    else:
        model_class = AlbertForDebias
        tokenizer_class = AlbertTokenizer

    # get tokenizer regardless of model version
    tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
    if model_args.model_name == "bert":
        if data_args.select_debiasing_prompts:
            model = model_class.from_pretrained("bert-base-uncased")
            model.bert = BertModel.from_pretrained(model_name_or_path)
        else:
            model = model_class.from_pretrained(model_name_or_path)
    elif model_args.model_name == "roberta":
        if data_args.select_debiasing_prompts:
            model = model_class.from_pretrained("roberta-base")
            model.roberta = RobertaModel.from_pretrained(model_name_or_path)
        else:
            model = model_class.from_pretrained(model_name_or_path)
    else:
        if data_args.select_debiasing_prompts:
            model = model_class.from_pretrained("albert-base-v2")
            model.albert = AlbertModel.from_pretrained(model_name_or_path)
        else:
            model = model_class.from_pretrained(model_name_or_path)

    # set DDP
    model.cuda().eval()
    model = amp.initialize(model)
    model = DataParallel(model)

    return model, tokenizer


def load_words(path: str, word_type: str) -> List[str]:
    if word_type == "wiki":
        _words = []
        with open(file=path, mode="r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                _words.append(lines[i].strip().split(sep=" ")[0])

    elif word_type == "stereotype":
        _words = []
        with open(file=path, mode="r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                _words.append(lines[i].strip())

    return _words


def clear_words(_words: List[str], tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]) -> List[str]:
    """Remove the input word if the word contains the out-of-vocabulary token.

    Args:
        _words (List[str]): Input words to check the out-of-vocabulary.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    words = []
    for i in range(len(_words)):
        if tokenizer.unk_token not in tokenizer.tokenize(_words[i]):
            words.append(_words[i])

    return words


class JSDivergence(nn.Module):
    def __init__(self, dim: int = 1, reduction: str = "batchmean") -> None:
        """Get average JS-Divergence between two networks.

        Args:
            dim (int, optional): A dimension along which softmax will be computed. Defaults to 1.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "batchmean".
        """
        super().__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, net1_logits: torch.FloatTensor, net2_logits: torch.FloatTensor) -> torch.FloatTensor:
        net1_dist = F.softmax(input=net1_logits, dim=self.dim)
        net2_dist = F.softmax(input=net2_logits, dim=self.dim)

        avg_dist = (net1_dist + net2_dist) / 2.0

        jsd_dist = 0.0
        jsd_dist += F.kl_div(input=F.log_softmax(net1_logits, dim=self.dim), target=avg_dist, reduction=self.reduction)
        jsd_dist += F.kl_div(input=F.log_softmax(net2_logits, dim=self.dim), target=avg_dist, reduction=self.reduction)

        return jsd_dist / 2.0


def get_indv_prompt_tokens(
    prompts: List[str],
    targ1_word: str,
    targ2_word: str,
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[BatchEncoding, BatchEncoding, np.ndarray, np.ndarray]:
    """Create a prompt with i-th target concept word and tokenize them.

    Args:
        prompts (List[str]): A total prompt words.
        targ1_word (str): An i-th target 1 word.
        targ2_word (str): An i-th target 2 word.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    targ1_sents = []
    targ2_sents = []
    # make targ1 and targ2 as sentences
    for i in range(len(prompts)):
        targ1_sents.append(targ1_word + " " + prompts[i] + " " + tokenizer.mask_token)
        targ2_sents.append(targ2_word + " " + prompts[i] + " " + tokenizer.mask_token)
        # targ1_sents.append(prompts[i] + " " + tokenizer.sep_token + " " + targ1_word + " " + tokenizer.mask_token + ".")
        # targ2_sents.append(prompts[i] + " " + tokenizer.sep_token + " " + targ2_word + " " + tokenizer.mask_token + ".")

    # tokenize targ1 and targ2 sentences
    targ1_tokens = tokenizer(text=targ1_sents, padding=True, truncation=True, return_tensors="pt")
    targ2_tokens = tokenizer(text=targ2_sents, padding=True, truncation=True, return_tensors="pt")

    # get mask token index
    targ1_mask_idx = np.where(torch.Tensor.numpy(targ1_tokens["input_ids"]) == tokenizer.mask_token_id)[1]
    targ2_mask_idx = np.where(torch.Tensor.numpy(targ2_tokens["input_ids"]) == tokenizer.mask_token_id)[1]

    return targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx


def to_cuda(targ1_tokens: BatchEncoding, targ2_tokens: BatchEncoding) -> Tuple[BatchEncoding, BatchEncoding]:
    for key in targ1_tokens.keys():
        targ1_tokens[key] = torch.Tensor.cuda(targ1_tokens[key])
        targ2_tokens[key] = torch.Tensor.cuda(targ2_tokens[key])

    return targ1_tokens, targ2_tokens


def get_batch_inputs(
    batch_idx: int,
    targ1_tokens: BatchEncoding,
    targ2_tokens: BatchEncoding,
    targ1_mask_idx: np.ndarray,
    targ2_mask_idx: np.ndarray,
    train_args: TrainingArguments,
) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.LongTensor], np.ndarray, np.ndarray]:
    """Slice all inputs as `batch_size`.

    Args:
        idx (int): An index for batch.
        targ1_tokens (BatchEncoding): Tokens for target 1 concepts.
        targ2_tokens (BatchEncoding): Tokens for target 2 concepts.
        targ1_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens.
        targ2_mask_idx (np.ndarray): Positions for [MASK] token in target 2 concept tokens.
        train_args (TrainingArguments): A parsed training arguments.

    Returns:
        targ1_inputs (Dict[str, torch.LongTensor]): Tokens for target 1 concepts sliced as `batch_size`.
        targ2_inputs (Dict[str, torch.LongTensor]): Tokens for target 2 concepts sliced as `batch_size`.
        targ1_local_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens sliced as `batch_size`.
        targ2_local_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens sliced as `batch_size`.
    """
    targ1_inputs = {}
    targ2_inputs = {}

    try:
        for key in targ1_tokens.keys():
            # slice to batch size
            targ1_inputs[key] = targ1_tokens[key][
                train_args.per_device_batch_size * batch_idx : train_args.per_device_batch_size * (batch_idx + 1)
            ]
            targ2_inputs[key] = targ2_tokens[key][
                train_args.per_device_batch_size * batch_idx : train_args.per_device_batch_size * (batch_idx + 1)
            ]

        targ1_batch_mask_idx = targ1_mask_idx[
            train_args.per_device_batch_size * batch_idx : train_args.per_device_batch_size * (batch_idx + 1)
        ]
        targ2_batch_mask_idx = targ2_mask_idx[
            train_args.per_device_batch_size * batch_idx : train_args.per_device_batch_size * (batch_idx + 1)
        ]

    except IndexError:
        for key in targ1_tokens.keys():
            # get rest of batches
            targ1_inputs[key] = targ1_tokens[key][train_args.per_device_batch_size * (batch_idx + 1) :]
            targ2_inputs[key] = targ2_tokens[key][train_args.per_device_batch_size * (batch_idx + 1) :]

        targ1_batch_mask_idx = targ1_mask_idx[train_args.per_device_batch_size * (batch_idx + 1) :]
        targ2_batch_mask_idx = targ2_mask_idx[train_args.per_device_batch_size * (batch_idx + 1) :]

    return targ1_inputs, targ2_inputs, targ1_batch_mask_idx, targ2_batch_mask_idx


def get_logits(
    model: Union[BertForDebias, RobertaForDebias, AlbertForDebias],
    inputs: Dict[str, torch.LongTensor],
    mask_idx: np.ndarray,
    stereotype_ids: List[int],
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Get logits corresponding to stereotype words at [MASK] token position.

    Args:
        model (Union[BertForAutoDebias, RobertaForDebias, AlbertForDebias]): A pre-trained language model for freezing.
        inputs (Dict[str, torch.LongTensor]): Tokenized prompt inputs with a [MASK] token.
        mask_idx (np.ndarray): An index of a [MASK] token in tokenized prompt inputs.
        stereotype_ids (List[int]): Pre-defined stereotype token ids.
    """
    outputs = model.forward(**inputs)
    pooler_output = outputs.pooler_output
    prediction_scores = outputs.prediction_scores

    # for [CLS]
    cls_logits = pooler_output[np.arange(torch.Tensor.size(inputs["input_ids"])[0]), :]
    # for [MASK] of stereotype ids
    mask_logits = prediction_scores[np.arange(torch.Tensor.size(inputs["input_ids"])[0]), mask_idx][:, stereotype_ids]

    return cls_logits, mask_logits


def get_jsd_values(
    targ1_tokens: BatchEncoding,
    targ2_tokens: BatchEncoding,
    targ1_mask_idx: np.ndarray,
    targ2_mask_idx: np.ndarray,
    model: Union[BertForDebias, RobertaForDebias, AlbertForDebias],
    stereotype_ids: List[int],
    jsd_module: JSDivergence,
    train_args: TrainingArguments,
) -> List[np.ndarray]:
    """Calculate JS-Divergence values and accumulate them for all prompts of i-th target concept word.

    Args:
        targ1_tokens (BatchEncoding): Tokens for target 1 concepts.
        targ2_tokens (BatchEncoding): Tokens for target 2 concepts.
        targ1_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens.
        targ2_mask_idx (np.ndarray): Positions for [MASK] token in target 2 concept tokens.
        model (Union[BertForDebias, RobertaForDebias, AlbertForDebias]): A pre-trained language model.
        stereotype_ids (List[int]): Pre-defined stereotype ids.
        jsd_module (JSDivergence): A JS-Divergence module for [MASK] token.
        train_args (TrainingArguments): A parsed training arguments.
    """
    jsd_values = []

    for batch_idx in range(torch.Tensor.size(targ1_tokens["input_ids"])[0] // train_args.per_device_batch_size + 1):
        # slice inputs as batch size
        targ1_inputs, targ2_inputs, targ1_batch_mask_idx, targ2_batch_mask_idx = get_batch_inputs(
            batch_idx=batch_idx,
            targ1_tokens=targ1_tokens,
            targ2_tokens=targ2_tokens,
            targ1_mask_idx=targ1_mask_idx,
            targ2_mask_idx=targ2_mask_idx,
            train_args=train_args,
        )

        # get [MASK] token logits of stereotype words
        _, targ1_mask_logits = get_logits(
            model=model, inputs=targ1_inputs, mask_idx=targ1_batch_mask_idx, stereotype_ids=stereotype_ids
        )
        _, targ2_mask_logits = get_logits(
            model=model, inputs=targ2_inputs, mask_idx=targ2_batch_mask_idx, stereotype_ids=stereotype_ids
        )

        # get JSD value for two networks
        # cls_jsd_dist = jsd_module.forward(net1_logits=targ1_cls_logits, net2_logits=targ2_cls_logits)
        mask_jsd_dist = jsd_module.forward(net1_logits=targ1_mask_logits, net2_logits=targ2_mask_logits)

        # get summed JSD for matching dimensions
        # cls_jsd_sum = np.sum(cls_jsd_dist.detach().cpu().numpy(), axis=1)
        mask_jsd_sum = np.sum(mask_jsd_dist.detach().cpu().numpy(), axis=1)

        # accumulate all JSD values
        jsd_values += list(mask_jsd_sum)

        del targ1_mask_logits, targ2_mask_logits, mask_jsd_dist, mask_jsd_sum

    return jsd_values


def get_accum_prompt_jsd_values(
    prompts: List[str],
    targ1_words: List[str],
    targ2_words: List[str],
    model: Union[BertForDebias, RobertaForDebias, AlbertForDebias],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    stereotype_ids: List[int],
    jsd_module: JSDivergence,
    train_args: TrainingArguments,
) -> np.ndarray:
    """Get JS-Divergence values for all prompts of all target concept words about bias.

    Args:
        prompts (List[str]): Candidate words for prompts.
        targ1_words (List[str]): Words for target 1 concepts.
        targ2_words (List[str]): Words for target 2 concepts.
        model (Union[BertForDebias, RobertaForDebias, AlbertForDebias]): A pre-trained freezing language model.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
        stereotype_ids (List[int]): Pre-defined stereotype ids.
        jsd_module (JSDivergence): A JS-Divergence module for [MASK] token.
        train_args (TrainingArguments): A parsed training arguments.

    Returns:
        accum_prompt_jsd_values (np.ndarray): Accumulated JS-Divergence values for all prompts.
    """
    prompt_jsd_values = []
    for i in tqdm(range(len(targ1_words))):
        # create all possible prompts combination for i-th target concept word
        targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx = get_indv_prompt_tokens(
            prompts=prompts, targ1_word=targ1_words[i], targ2_word=targ2_words[i], tokenizer=tokenizer
        )

        # send all tokens to cuda for GPU calculating
        targ1_tokens, targ2_tokens = to_cuda(targ1_tokens=targ1_tokens, targ2_tokens=targ2_tokens)

        # get JS-Divergence values of i-th target concept word
        jsd_values = get_jsd_values(
            targ1_tokens=targ1_tokens,
            targ2_tokens=targ2_tokens,
            targ1_mask_idx=targ1_mask_idx,
            targ2_mask_idx=targ2_mask_idx,
            model=model,
            stereotype_ids=stereotype_ids,
            jsd_module=jsd_module,
            train_args=train_args,
        )
        # accumulate all target concept words
        prompt_jsd_values.append(jsd_values)
    prompt_jsd_values = np.array(prompt_jsd_values)
    accum_prompt_jsd_values = np.mean(prompt_jsd_values, axis=0)

    return accum_prompt_jsd_values
