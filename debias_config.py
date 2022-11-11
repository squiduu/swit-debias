from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them.
    """

    data_dir: Optional[str] = field(default=None, metadata={"help": "A directory path of input data."})
    debias_type: Optional[str] = field(default=None, metadata={"help": "A type of debias."})
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    divergence_dim: Optional[int] = field(
        default=1, metadata={"help": "A dimension along which axis will be computed."}
    )
    prompt_dir: Optional[str] = field(default=None, metadata={"help": "The prompt directory to debias."})


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class TrainingArguments:
    """The arguments we use in our scripts which relate to the training loop itself."""

    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    num_gpus: Optional[int] = field(default=1, metadata={"help": "The number of GPUs for training."})
    world_size: Optional[int] = field(default=1, metadata={"help": "The number of total GPUs for training."})
    per_device_batch_size: Optional[int] = field(default=1, metadata={"help": "The number of batch size per device."})
    local_rank: Optional[int] = field(default=0, metadata={"help": "??."})
    distributed: Optional[bool] = field(default=False, metadata={"help": "Whether or not to use distributed training."})
    dataloader_num_workers: Optional[int] = field(default=0, metadata={"help": "The number of dataloader workers."})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "The value of learning rate."})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "The number of training epochs."})
    precision: Optional[int] = field(default=16, metadata={"help": "The bit number of precision level."})
    seed: Optional[int] = field(default=42, metadata={"help": "The seed number for initialization."})
