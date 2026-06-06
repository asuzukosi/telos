"""mini-swe docker/local environment factory for swe-bench instances."""

from __future__ import annotations

import subprocess
from typing import Any

from agenticml.evaluation.benchmarks.swe.common import ensure_miniswe_on_path
from agenticml.evaluation.benchmarks.swe.registry import BashEnvironment

DEFAULT_MAX_ITERATIONS = 250
REPEAT_COMMAND_LIMIT = 3
DEFAULT_PULL_TIMEOUT = 600


def docker_image_name(instance: dict[str, Any]) -> str:
    ensure_miniswe_on_path()
    from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

    return get_swebench_docker_image_name(instance)


def pull_instance_image(instance: dict[str, Any], *, timeout: int | None = None) -> str:
    """pull one instance eval image; return image ref."""
    image = docker_image_name(instance)
    subprocess.run(["docker", "pull", image], check=True, timeout=timeout)
    return image


def default_swe_config() -> dict[str, Any]:
    """load upstream swebench.yaml agent/environment defaults."""
    ensure_miniswe_on_path()
    from minisweagent.config import builtin_config_dir, get_config_from_spec

    path = builtin_config_dir / "benchmarks" / "swebench.yaml"
    cfg = get_config_from_spec(str(path))
    cfg.setdefault("environment", {})["pull_timeout"] = DEFAULT_PULL_TIMEOUT
    return cfg


def environment_for_instance(
    instance: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    environment_class: str | None = None,
) -> BashEnvironment:
    """start mini-swe environment for one swe-bench instance (docker by default)."""
    ensure_miniswe_on_path()
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    cfg: dict[str, Any] = dict(config or default_swe_config())
    if environment_class:
        cfg.setdefault("environment", {})["environment_class"] = environment_class
    iid = str(instance.get("instance_id", "?"))
    image = docker_image_name(instance)
    try:
        return get_sb_environment(cfg, instance)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"swe docker start timed out for {iid} after {exc.timeout}s "
            f"(image pull/start hung). fix containerd/docker, then pre-pull:\n"
            f"  docker pull {image}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        msg = f"swe docker start failed for {iid} (exit {exc.returncode})"
        if detail:
            msg = f"{msg}: {detail}"
        raise RuntimeError(msg) from exc


def cleanup_environment(env: BashEnvironment) -> None:
    cleanup = getattr(env, "cleanup", None)
    if callable(cleanup):
        cleanup()
