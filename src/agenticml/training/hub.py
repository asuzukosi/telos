from __future__ import annotations

from agenticml.training.types import TrainingMode


def maybe_push_artifacts(
    *,
    model,
    tokenizer,
    hub_repo_id: str | None = None,
    mode: TrainingMode = TrainingMode.LORA,
) -> None:
    if not hub_repo_id:
        return
    if mode is TrainingMode.LORA:
        adapter_repo_id = f"{hub_repo_id}-adapter"
        print(f"pushing lora adapter to {adapter_repo_id}...")
        model.push_to_hub(adapter_repo_id)
        tokenizer.push_to_hub(adapter_repo_id)
        print("merging adapters...")
        model = model.merge_and_unload()
    print(f"pushing weights to {hub_repo_id}...")
    model.push_to_hub(hub_repo_id)
    tokenizer.push_to_hub(hub_repo_id)
