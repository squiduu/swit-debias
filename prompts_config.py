from dataclasses import dataclass, field
from typing import Optional

MALE_WORDS = [
    "fathers",
    "actor",
    "prince",
    "men",
    "gentlemen",
    "sir",
    "brother",
    "his",
    "king",
    "husband",
    "dad",
    "males",
    "sir",
    "him",
    "boyfriend",
    "he",
    "hero",
    "kings",
    "brothers",
    "son",
    "sons",
    "himself",
    "gentleman",
    "his",
    "father",
    "male",
    "man",
    "grandpa",
    "boy",
    "grandfather",
]
FEMALE_WORDS = [
    "mothers",
    "actress",
    "princess",
    "women",
    "ladies",
    "madam",
    "sister",
    "her",
    "queen",
    "wife",
    "mom",
    "females",
    "miss",
    "her",
    "girlfriend",
    "she",
    "heroine",
    "queens",
    "sisters",
    "daughter",
    "daughters",
    "herself",
    "lady",
    "hers",
    "mother",
    "female",
    "woman",
    "grandma",
    "girl",
    "grandmother",
]
AFA_WORDS = [
    "black",
    "african",
    "black",
    "africa",
    "africa",
    "africa",
    "black people",
    "african people",
    "black people",
    "the africa",
]
EUA_WORDS = [
    "caucasian",
    "caucasian",
    "white",
    "america",
    "america",
    "europe",
    "caucasian people",
    "caucasian people",
    "white people",
    "the america",
]


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
    max_prompt_length: int = field(default=5, metadata={"help": "The maximum value of biased prompt length."})
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
    num_p: int = field(default=1, metadata={"help": "The maximum number of selected prompts."})
    jsd_dimension: Optional[int] = field(default=1, metadata={"help": "A dimension along which axis will be computed."})
    select_biased_prompts: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to select top prompts."}
    )
    select_debiasing_prompts: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to select bottom prompts."}
    )
    prompt_dir: Optional[str] = field(default=None, metadata={"help": "The prompt directory to debias."})


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name: str = field(
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
    """It is the subset of the arguments we use in our example scripts which relate to the training loop itself."""

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
    seed: Optional[int] = field(default=42, metadata={"help": "A seed number."})
