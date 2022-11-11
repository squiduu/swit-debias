from logging import Logger
from torch.optim.adamw import AdamW
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import set_seed
from debias_config import DataArguments, ModelArguments, TrainingArguments
from debias_utils import (
    clear_console,
    get_logger,
    get_logits,
    prepare_model_and_tokenizer,
    JSDivergence,
    save_checkpoint,
    AverageKLDivergence,
)
from debias_dataloader import read_data, get_dataloaders


def run_debias(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments, logger: Logger):
    """Remove bias with generated biased prompts.

    Args:
        data_args (DataArguments): A parsed data arguments.
        model_args (ModelArguments): A parsed model arguments.
        train_args (TrainingArguments): A parsed training arguments.
        logger (Logger): A logger for checking progress.
    """
    logger.info(f"Model args: {vars(model_args)}")
    logger.info(f"Data args: {vars(data_args)}")
    logger.info(f"Train args: {vars(train_args)}")

    logger.info(f"Set seed as {train_args.seed}.")
    set_seed(train_args.seed)

    logger.info("Prepare models and tokenizer.")
    fixed_model, tuning_model, tokenizer = prepare_model_and_tokenizer(model_name_or_path=model_args.model_name_or_path)

    logger.info("Prepare a dataloader.")
    data = read_data(tokenizer=tokenizer, data_args=data_args, logger=logger)
    dataloaders = get_dataloaders(
        data=data, data_args=data_args, model_args=model_args, train_args=train_args, logger=logger
    )

    logger.info("Set loss function and optimizers for models.")
    jsd_criterion = JSDivergence(dim=data_args.divergence_dim, reduction="batchmean")
    kld_criterion = AverageKLDivergence(dim=data_args.divergence_dim, reduction="batchmean")
    optimizer = AdamW(params=tuning_model.parameters(), lr=train_args.learning_rate)

    logger.info("Start to fine-tune.")
    logger.info(f"Total batch size: {train_args.per_device_batch_size * train_args.num_gpus}")
    for epoch in range(1, int(train_args.num_train_epochs) + 1):
        # init loss for an epoch
        epoch_loss = 0.0

        # load batch data
        for dataloader in dataloaders:
            # set gradients as zero
            optimizer.zero_grad()

            # get model output logits
            targ1_fixed_cls_logits, targ1_tuning_cls_logits, targ1_mask_logits = get_logits(
                fixed_model=fixed_model,
                tuning_model=tuning_model,
                inputs=dataloader["targ1_input_ids"],
                attention_mask=dataloader["targ1_attention_mask"],
                token_type_ids=dataloader["targ1_token_type_ids"],
                mask_idx=dataloader["targ1_mask_idx"],
            )
            targ2_fixed_cls_logits, targ2_tuning_cls_logits, targ2_mask_logits = get_logits(
                fixed_model=fixed_model,
                tuning_model=tuning_model,
                inputs=dataloader["targ2_input_ids"],
                attention_mask=dataloader["targ2_attention_mask"],
                token_type_ids=dataloader["targ2_token_type_ids"],
                mask_idx=dataloader["targ2_mask_idx"],
            )

            # get JSD and KLD for a [MASK] token of tuning model
            mask_jsd_loss = jsd_criterion.forward(net1_logits=targ1_mask_logits, net2_logits=targ2_mask_logits)
            mask_kld_loss = kld_criterion.forward(net1_logits=targ1_mask_logits, net2_logits=targ2_mask_logits)

            # get JSD and KLD for a [CLS] token between fixed and tuning models
            targ1_cls_jsd_loss = jsd_criterion.forward(
                net1_logits=targ1_fixed_cls_logits, net2_logits=targ1_tuning_cls_logits
            )
            targ2_cls_jsd_loss = jsd_criterion.forward(
                net1_logits=targ2_fixed_cls_logits, net2_logits=targ2_tuning_cls_logits
            )
            targ1_cls_kld_loss = kld_criterion.forward(
                net1_logits=targ1_fixed_cls_logits, net2_logits=targ1_tuning_cls_logits
            )
            targ2_cls_kld_loss = kld_criterion.forward(
                net1_logits=targ2_fixed_cls_logits, net2_logits=targ2_tuning_cls_logits
            )

            # get loss
            loss = (
                mask_jsd_loss
                + mask_kld_loss
                + targ1_cls_jsd_loss
                + targ2_cls_jsd_loss
                + targ1_cls_kld_loss
                + targ2_cls_kld_loss
            )

            # make loss smaller
            loss.backward()
            optimizer.step()

            if iter % 1000 == 0:
                logger.info(
                    f"Epoch: {epoch}/{int(train_args.num_train_epochs)} - Iter: {iter}/{len(dataloaders)} - Loss: {loss:.4f}"
                )

            # accumulate batch loss
            epoch_loss += loss

        # after an epoch
        logger.info(f"Epoch: {epoch}/{int(train_args.num_train_epochs)} - Loss: {epoch_loss / len(dataloaders):.4f}")

        logger.info("Save a debiased model for GLUE and SEAT.")
        save_checkpoint(
            tuning_model=tuning_model,
            tokenizer=tokenizer,
            epoch=epoch,
            data_args=data_args,
            model_args=model_args,
            train_args=train_args,
        )


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    logger = get_logger(train_args=train_args)

    run_debias(data_args=data_args, model_args=model_args, train_args=train_args, logger=logger)
