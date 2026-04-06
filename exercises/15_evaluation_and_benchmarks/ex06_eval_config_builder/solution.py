"""
Solution for Exercise 6: Build Evaluation Configurations
"""

from dataclasses import dataclass, field
from typing import Any

_MISSING = object()


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset."""
    name: str = ""
    path: str = ""
    rm_type: str | None = None
    input_key: str | None = None
    label_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_response_len: int | None = None
    n_samples: int | None = None
    metadata_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata_overrides is None:
            self.metadata_overrides = {}
        if not isinstance(self.metadata_overrides, dict):
            raise TypeError("metadata_overrides must be a dict")

    def inject_metadata(self, sample_metadata: Any) -> dict[str, Any]:
        """Return a new metadata dict with overrides applied."""
        if not isinstance(sample_metadata, dict):
            metadata = {}
        else:
            metadata = dict(sample_metadata)

        if self.rm_type is not None:
            metadata["rm_type"] = self.rm_type

        for key, value in self.metadata_overrides.items():
            metadata[key] = value

        return metadata


def resolve_field(
    dataset_value: Any,
    default_value: Any,
    global_value: Any,
) -> Any:
    """Resolve a config field using the priority hierarchy."""
    if dataset_value is not None:
        return dataset_value
    if default_value is not None:
        return default_value
    if global_value is not None:
        return global_value
    return None


def _get_global(global_args: dict, *keys: str) -> Any:
    """Get first non-None value from global_args by trying multiple keys."""
    if global_args is None:
        return None
    for key in keys:
        val = global_args.get(key)
        if val is not None:
            return val
    return None


def build_eval_configs(
    datasets_raw: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    global_args: dict[str, Any] | None = None,
) -> list[EvalDatasetConfig]:
    """Build a list of EvalDatasetConfig from raw dicts with resolution."""
    defaults = defaults or {}
    global_args = global_args or {}
    configs = []

    for raw in datasets_raw:
        if "name" not in raw:
            raise ValueError("Each dataset dict must include a 'name' field.")

        name = raw["name"]
        path = raw.get("path", "")
        rm_type = raw.get("rm_type")
        metadata_overrides = raw.get("metadata_overrides", {})

        temperature = resolve_field(
            raw.get("temperature"),
            defaults.get("temperature"),
            _get_global(global_args, "temperature"),
        )
        top_p = resolve_field(
            raw.get("top_p"),
            defaults.get("top_p"),
            _get_global(global_args, "top_p"),
        )
        max_response_len = resolve_field(
            raw.get("max_response_len"),
            defaults.get("max_response_len"),
            _get_global(global_args, "max_response_len", "rollout_max_response_len"),
        )
        n_samples = resolve_field(
            raw.get("n_samples"),
            defaults.get("n_samples"),
            _get_global(global_args, "n_samples", "n_samples_per_prompt"),
        )
        input_key = resolve_field(
            raw.get("input_key"),
            defaults.get("input_key"),
            _get_global(global_args, "input_key"),
        )
        label_key = resolve_field(
            raw.get("label_key"),
            defaults.get("label_key"),
            _get_global(global_args, "label_key"),
        )

        cfg = EvalDatasetConfig(
            name=name,
            path=path,
            rm_type=rm_type,
            input_key=input_key,
            label_key=label_key,
            temperature=temperature,
            top_p=top_p,
            max_response_len=max_response_len,
            n_samples=n_samples,
            metadata_overrides=metadata_overrides,
        )
        configs.append(cfg)

    return configs
