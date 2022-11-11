import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List
from logging import Logger
from torch.nn.parallel.data_parallel import DataParallel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel
from transformers.models.albert.modeling_albert import AlbertForSequenceClassification, AlbertModel
from modelings import AlbertForDebias, BertForDebias, RobertaForDebias
from debias_config import DataArguments, ModelArguments, TrainingArguments


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
        args (Namespace): A parsed arguments.

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
    file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"ada_{train_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run name: {train_args.run_name}")

    return logger


def prepare_model_and_tokenizer(
    model_name_or_path: str,
) -> Tuple[
    Union[BertForDebias, RobertaForDebias, AlbertForDebias],
    Union[BertForDebias, RobertaForDebias, AlbertForDebias],
    Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
]:
    """Download and prepare the pre-trained model and tokenizer.

    Args:
        model_name_or_path (str): A name of pre-trained model.
        run_type (str): A status of either 'prompt' or 'debias'.
    """
    if "bert" in model_name_or_path:
        model_class = BertForDebias
        tokenizer_class = BertTokenizer
    elif "roberta" in model_name_or_path:
        model_class = RobertaForDebias
        tokenizer_class = RobertaTokenizer
    else:
        model_class = AlbertForDebias
        tokenizer_class = AlbertTokenizer

    # get common tokenizer regardless of run type
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    fixed_model = model_class.from_pretrained(model_name_or_path)
    tuning_model = model_class.from_pretrained(model_name_or_path)

    fixed_model.cuda().eval()
    tuning_model.cuda().train()

    fixed_model = DataParallel(fixed_model)
    tuning_model = DataParallel(tuning_model)

    return fixed_model, tuning_model, tokenizer


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


class AverageKLDivergence(nn.Module):
    def __init__(self, dim: int = 1, reduction: str = "batchmean") -> None:
        """Get average KL-Divergence between two networks.

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

        kld_dist = 0.0
        kld_dist += F.kl_div(input=F.log_softmax(net1_logits, dim=self.dim), target=net2_dist, reduction=self.reduction)
        kld_dist += F.kl_div(input=F.log_softmax(net2_logits, dim=self.dim), target=net1_dist, reduction=self.reduction)

        return kld_dist / 2.0


def load_words(path: str) -> List[str]:
    _words = []
    with open(file=path, mode="r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            _words.append(lines[i].strip())

    return _words


def clear_words(
    _words1: List[str],
    _words2: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[List[str], List[str]]:
    """Remove the input word if the word contains the out-of-vocabulary token.

    Args:
        _words1 (List[str]): Input words to check the out-of-vocabulary.
        _words2 (List[str]): Input words to check the out-of-vocabulary.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    words1 = []
    words2 = []
    for i in range(len(_words1)):
        if tokenizer.unk_token not in (tokenizer.tokenize(_words1[i]) + tokenizer.tokenize(_words2[i])):
            words1.append(_words1[i])
            words2.append(_words2[i])

    return words1, words2


def load_sentences(path: str) -> List[str]:
    with open(file=path, mode="r") as f:
        lines = f.readlines()
        _sentence = list(set(lines[i].strip() for i in range(len(lines)) if 75 <= len(lines[i].strip()) <= 100))

    return _sentence


def clear_sentences(
    _sentence: List[str], tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]
) -> List[str]:
    """Remove the input word if the word contains the out-of-vocabulary token.

    Args:
        _sentence (List[str]): Input sentences to check the OOV.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    sentence = [
        _sentence[i] for i in range(len(_sentence)) if tokenizer.unk_token not in tokenizer.tokenize(_sentence[i])
    ]

    return sentence


def to_cuda(targ1_tokens: BatchEncoding, targ2_tokens: BatchEncoding) -> Tuple[BatchEncoding, BatchEncoding]:
    for key in targ1_tokens.keys():
        targ1_tokens[key] = torch.Tensor.cuda(targ1_tokens[key])
        targ2_tokens[key] = torch.Tensor.cuda(targ2_tokens[key])

    return targ1_tokens, targ2_tokens


def tokenize_prompts(
    prompts: List[str],
    targ1_words: List[str],
    targ2_words: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[BatchEncoding, BatchEncoding, np.ndarray, np.ndarray]:
    """Create prompts with target concept word and tokenize them.

    Args:
        prompts (List[str]): A searched prompt words.
        targ1_words (List[str]): An target 1 words.
        targ2_words (List[str]): An target 2 words.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    targ1_sents = []
    targ2_sents = []
    # make targ1 and targ2 sentences
    for i in range(len(prompts)):
        for j in range(len(targ1_words)):
            targ1_sents.append(targ1_words[j] + " " + prompts[i] + " " + tokenizer.mask_token + ".")
            targ2_sents.append(targ2_words[j] + " " + prompts[i] + " " + tokenizer.mask_token + ".")

    # tokenize targ1 and targ2 sentences
    targ1_tokens = tokenizer(text=targ1_sents, padding=True, truncation=True, return_tensors="pt")
    targ2_tokens = tokenizer(text=targ2_sents, padding=True, truncation=True, return_tensors="pt")

    # get mask token index
    targ1_mask_idx = np.where(torch.Tensor.numpy(targ1_tokens["input_ids"]) == tokenizer.mask_token_id)[1]
    targ2_mask_idx = np.where(torch.Tensor.numpy(targ2_tokens["input_ids"]) == tokenizer.mask_token_id)[1]

    return targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx


def get_logits(
    fixed_model: Union[BertForDebias, RobertaForDebias, AlbertForDebias],
    tuning_model: Union[BertForDebias, RobertaForDebias, AlbertForDebias],
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    token_type_ids: torch.LongTensor,
    mask_idx: np.ndarray,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Get the last hidden states of input sequence and logits on [MASK] token position.

    Args:
        freezing_model (Union[BertForDebias, RobertaForDebias, AlbertForDebias]): A pre-trained language model for freezing.
        tuning_model (Union[BertForDebias, RobertaForDebias, AlbertForDebias]): A pre-trained language model for fine-tuning.
        inputs (Dict[str, torch.LongTensor]): Tokenized prompt inputs with a [MASK] token.
        mask_idx (np.ndarray): An index of a [MASK] token in tokenized prompt inputs.
    """
    fixed_outputs = fixed_model.forward(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
    )
    tuning_outputs = tuning_model.forward(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
    )

    # get [MASK] output for tuning model
    mask_logits = tuning_outputs.prediction_scores[torch.arange(tuning_outputs.prediction_scores.size()[0]), mask_idx]

    return fixed_outputs.pooler_output, tuning_outputs.pooler_output, mask_logits


def get_cosine_similarity(logits1: torch.FloatTensor, logits2: torch.FloatTensor) -> torch.FloatTensor:
    cos_sim = F.cosine_similarity(logits1, logits2)

    return cos_sim.mean()


def overwrite_state_dict(
    tuning_model: DataParallel, model_args: ModelArguments
) -> Tuple[
    Union[BertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification],
    Union[BertModel, RobertaModel, AlbertModel],
]:
    """Extract and transfer only the trained weights of the layer matching the new model.

    Args:
        trained_model (DebiasRunner): A debiased fine-tuned model.
        model_args (ModelArguments): A parsed model arguments.
    """
    if "bert" in model_args.model_name_or_path:
        glue_model_class = BertForSequenceClassification
        seat_model_class = BertModel
    elif "roberta" in model_args.model_name_or_path:
        glue_model_class = RobertaForSequenceClassification
        seat_model_class = RobertaModel
    else:
        glue_model_class = AlbertForSequenceClassification
        seat_model_class = AlbertModel

    # get initialized pre-trained model
    glue_model = glue_model_class.from_pretrained(model_args.model_name_or_path)
    seat_model = seat_model_class.from_pretrained(model_args.model_name_or_path)

    # get only state dict to move to new models
    trained_state_dict = tuning_model.module.state_dict()
    glue_state_dict = glue_model.state_dict()
    seat_state_dict = seat_model.state_dict()

    new_glue_state_dict = {k: v for k, v in trained_state_dict.items() if k in glue_state_dict}
    if "bert" in model_args.model_name_or_path:
        new_seat_state_dict = {k[5:]: v for k, v in trained_state_dict.items() if k[5:] in seat_state_dict}
    elif "roberta" in model_args.model_name_or_path:
        new_seat_state_dict = {k[8:]: v for k, v in trained_state_dict.items() if k[8:] in seat_state_dict}
    else:
        new_seat_state_dict = {k[7:]: v for k, v in trained_state_dict.items() if k[7:] in seat_state_dict}

    # overwrite entries in the existing initialized state dict
    glue_state_dict.update(new_glue_state_dict)
    seat_state_dict.update(new_seat_state_dict)

    # overwrite updated weights
    glue_model.load_state_dict(glue_state_dict)
    seat_model.load_state_dict(seat_state_dict)

    return glue_model, seat_model


def save_checkpoint(
    tuning_model: Union[BertModel, RobertaModel, AlbertModel],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    data_args: DataArguments,
    model_args: ModelArguments,
    train_args: TrainingArguments,
):
    # get state dict for glue and seat
    glue_model, seat_model = overwrite_state_dict(tuning_model=tuning_model, model_args=model_args)

    # save for glue
    glue_model.save_pretrained(
        train_args.output_dir + f"glue_{model_args.model_name_or_path}_{train_args.run_name}_{data_args.debias_type}"
    )
    tokenizer.save_pretrained(
        train_args.output_dir + f"glue_{model_args.model_name_or_path}_{train_args.run_name}_{data_args.debias_type}"
    )

    # save for seat
    seat_model.save_pretrained(
        train_args.output_dir + f"seat_{model_args.model_name_or_path}_{train_args.run_name}_{data_args.debias_type}"
    )
    tokenizer.save_pretrained(
        train_args.output_dir + f"seat_{model_args.model_name_or_path}_{train_args.run_name}_{data_args.debias_type}"
    )
