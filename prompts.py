import torch
import torch.distributed as dist
import numpy as np
import random
from io import TextIOWrapper
from logging import Logger
from pytorch_lightning.utilities.seed import seed_everything
from transformers.hf_argparser import HfArgumentParser
from prompts_config import (
    MALE_WORDS,
    FEMALE_WORDS,
    AFA_WORDS,
    EUA_WORDS,
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from prompts_utils import (
    clear_console,
    get_logger,
    prepare_model_and_tokenizer,
    load_words,
    clear_words,
    JSDivergence,
    get_accum_prompt_jsd_values,
)


def generate_prompts(
    data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger
):
    """Generate prompts and save them using JS-Divergence.

    Args:
        data_args (DataArguments): A parsed data arguments.
        model_args (ModelArguments): A parsed model arguments.
        train_args (TrainingArguments): A parsed training arguments.
        logger (Logger): A logger for checking progress information.
    """
    logger.info(f"Data args: {data_args}")
    logger.info(f"Model args: {model_args}")
    logger.info(f"Training args: {train_args}")

    logger.info("Set distributed data parallel training.")
    torch.cuda.set_device(train_args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    train_args.world_size = dist.get_world_size()

    seed = random.randint(0, 100)
    logger.info(f"Set seed: {seed}")
    seed_everything(seed)

    # prepare pre-trained model and tokenizer
    logger.info(f"Prepare pre-trained model and tokenizer: {model_args.model_name_or_path}")
    model, tokenizer = prepare_model_and_tokenizer(
        model_name_or_path=model_args.model_name_or_path, data_args=data_args, model_args=model_args
    )

    # load and tokenize stereotype words
    logger.info(f"Load and tokenize stereotype words: {data_args.data_dir + 'stereotype_words.txt'}")
    _words = load_words(path=data_args.data_dir + "stereotype_words.txt", word_type="stereotype")
    logger.info("Remove words containing OOV tokens in stereotype.")
    stereotype_words = clear_words(_words=_words, tokenizer=tokenizer)
    STEREOTYPE_IDS = tokenizer.convert_tokens_to_ids(stereotype_words)

    # load wiki words
    logger.info(f"Load wiki words: {data_args.data_dir + 'wiki_words_5000.txt'}")
    _words = load_words(path=data_args.data_dir + "wiki_words_5000.txt", word_type="wiki")
    logger.info("Remove words containing OOV tokens in wiki.")
    WIKI_WORDS = clear_words(_words=_words, tokenizer=tokenizer)

    # init prompts
    current_prompts = WIKI_WORDS

    # create and open prompt file
    logger.info(f"Create and open prompt file.")
    prompt_file: TextIOWrapper = open(file=data_args.prompt_dir, mode="w")

    # init js-divergence
    jsd_module = JSDivergence(dim=data_args.jsd_dimension, reduction="none")

    # calculate js-divergence for prompts
    logger.info(f"Get JS-Divergence values for all prompts about {data_args.debias_type} bias.")
    for i in range(data_args.max_prompt_length):
        logger.info(f"Maximum prompt length: {i + 1}")
        logger.info(f"Number of prompts: {len(current_prompts)}")

        if data_args.debias_type == "gender":
            accum_prompt_jsd_values = get_accum_prompt_jsd_values(
                prompts=current_prompts,
                targ1_words=MALE_WORDS,
                targ2_words=FEMALE_WORDS,
                model=model,
                tokenizer=tokenizer,
                stereotype_ids=STEREOTYPE_IDS,
                jsd_module=jsd_module,
                train_args=train_args,
            )

        elif data_args.debias_type == "race":
            accum_prompt_jsd_values = get_accum_prompt_jsd_values(
                prompts=current_prompts,
                targ1_words=AFA_WORDS,
                targ2_words=EUA_WORDS,
                model=model,
                tokenizer=tokenizer,
                stereotype_ids=STEREOTYPE_IDS,
                jsd_module=jsd_module,
                train_args=train_args,
            )

        logger.info(f"Select prompts.")
        if data_args.select_biased_prompts:
            biased_prompts = np.array(current_prompts)[np.argsort(accum_prompt_jsd_values)[::-1][: data_args.num_p]]
        else:
            biased_prompts = np.array([])

        if data_args.select_debiasing_prompts:
            debiasing_prompts = np.array(current_prompts)[np.argsort(accum_prompt_jsd_values)[: data_args.num_p]]
        else:
            debiasing_prompts = np.array([])

        selected_prompts = np.append(biased_prompts, debiasing_prompts)

        logger.info(f"Write selected prompts to the file.")
        for selcted_prompt in selected_prompts:
            prompt_file.write(selcted_prompt)
            prompt_file.write("\n")

        logger.info("Create temporary prompts.")
        temp_prompts = []
        for selcted_prompt in selected_prompts:
            for wiki_word in WIKI_WORDS:
                temp_prompts.append(selcted_prompt + " " + wiki_word)

        logger.info("Update current prompts.")
        current_prompts = temp_prompts

    logger.info(f"Save and close prompt file.")
    prompt_file.close()


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args)

    generate_prompts(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
