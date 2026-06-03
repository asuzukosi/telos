from __future__ import annotations

from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer, Trainer

from telos.training.lora_common import (
    RunConfig,
    build_lora_model,
    causal_lm_collator,
    make_training_args,
    maybe_init_wandb,
    maybe_push_artifacts,
    print_trainable,
    set_seed,
    tokenize_chatml_data_for_training,
)


def run_chatml_lora_train(
    cfg: RunConfig,
    *,
    tokenizer_id: str = "",
    limit_train: int = 0,
    limit_eval: int = 0,
    adapter_repo_id: str | None = None,
    merged_repo_id: str | None = None,
) -> None:
    set_seed(cfg.seed)
    maybe_init_wandb(cfg)

    tok_id = tokenizer_id or cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_lora_model(cfg.model_id)
    print_trainable(model)

    ds = load_dataset(cfg.dataset_id)
    train_ds = ds[cfg.train_split]
    eval_ds = ds[cfg.eval_split]
    if limit_train:
        train_ds = train_ds.select(range(min(limit_train, len(train_ds))))
    if limit_eval:
        eval_ds = eval_ds.select(range(min(limit_eval, len(eval_ds))))

    tokenize = partial(
        tokenize_chatml_data_for_training,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    train_tok = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    train_tok = train_tok.filter(lambda x: len(x["input_ids"]) > 0)

    eval_tok = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)
    eval_tok = eval_tok.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"chatml rows kept: train={len(train_tok)} eval={len(eval_tok)}")

    targs = make_training_args(cfg)
    collator = partial(causal_lm_collator, pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    trainer.save_state()
    maybe_push_artifacts(
        model=model,
        tokenizer=tokenizer,
        adapter_repo_id=adapter_repo_id,
        merged_repo_id=merged_repo_id,
    )
