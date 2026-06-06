from __future__ import annotations

from functools import partial

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, Trainer

from agenticml.training.collator import causal_lm_collator
from agenticml.training.hub import maybe_push_artifacts
from agenticml.training.model import build_model, print_trainable
from agenticml.training.setup import make_training_args, maybe_init_wandb, set_seed
from agenticml.training.tokenize import tokenize_data_for_training
from agenticml.training.types import RunConfig, TrainingPromptField


def run_supervised_train(
    cfg: RunConfig,
    *,
    prompt_field: TrainingPromptField,
    limit_train: int = 0,
    limit_eval: int = 0,
    hub_repo_id: str | None = None,
) -> None:
    set_seed(cfg.seed)
    maybe_init_wandb(cfg)

    print(f"loading tokenizer from {cfg.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"building model ({cfg.mode.value}) from {cfg.model_id}...")
    model = build_model(cfg.model_id, cfg.mode)
    print_trainable(model)

    ds = load_dataset(cfg.dataset_id)
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"expected a split dataset (DatasetDict), got {type(ds).__name__}")
    train_ds = ds[cfg.train_split]
    eval_ds = ds[cfg.eval_split]
    if limit_train:
        train_ds = train_ds.select(range(min(limit_train, len(train_ds))))
    if limit_eval:
        eval_ds = eval_ds.select(range(min(limit_eval, len(eval_ds))))

    tokenize = partial(
        tokenize_data_for_training,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        prompt_field=prompt_field,
    )
    train_tok = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    train_tok = train_tok.filter(lambda x: len(x["input_ids"]) > 0)
    eval_tok = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)
    eval_tok = eval_tok.filter(lambda x: len(x["input_ids"]) > 0)
    print(
        f"{prompt_field.value} rows kept ({cfg.mode.value}): "
        f"train={len(train_tok)} eval={len(eval_tok)}"
    )

    targs = make_training_args(cfg)
    collator = partial(causal_lm_collator, pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    trainer.save_state()
    maybe_push_artifacts(
        model=model,
        tokenizer=tokenizer,
        hub_repo_id=hub_repo_id,
        mode=cfg.mode,
    )
