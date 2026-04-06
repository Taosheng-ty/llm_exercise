"""
Exercise 6: Build Evaluation Configurations

In large-scale LLM evaluation, different datasets need different settings
(temperature, max tokens, input keys, etc.). A configuration system must
support per-dataset overrides with a resolution hierarchy:

    dataset-specific -> default -> global fallback

Reference: slime/utils/eval_config.py EvalDatasetConfig and
build_eval_dataset_configs().

Your task: implement EvalDatasetConfig, resolve_field(), and
build_eval_configs().

Difficulty: Medium
Framework: numpy / stdlib
"""

from dataclasses import dataclass, field
from typing import Any

# Sentinel for missing values
_MISSING = object()


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset.

    Attributes:
        name: Dataset identifier.
        path: Path to the dataset file.
        rm_type: Reward model type (e.g., "gpqa", "math", "f1").
        input_key: Key for input text in the dataset.
        label_key: Key for label in the dataset.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_response_len: Maximum response length.
        n_samples: Number of samples per prompt.
        metadata_overrides: Extra key-value pairs to inject into sample metadata.

    The metadata_overrides field should default to an empty dict.
    In __post_init__, validate that metadata_overrides is a dict (raise
    TypeError if not).
    """
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
        # TODO: Validate metadata_overrides is a dict
        raise NotImplementedError("Implement __post_init__")

    def inject_metadata(self, sample_metadata: Any) -> dict[str, Any]:
        """Return a new metadata dict with overrides applied.

        If sample_metadata is not a dict, start with empty dict.
        Then apply all metadata_overrides on top.
        If rm_type is set, also inject it as metadata["rm_type"].

        Args:
            sample_metadata: Original sample metadata (possibly None or non-dict).

        Returns:
            New dict with overrides applied.
        """
        # TODO: Implement this method
        raise NotImplementedError("Implement inject_metadata")


def resolve_field(
    dataset_value: Any,
    default_value: Any,
    global_value: Any,
) -> Any:
    """Resolve a config field using the priority hierarchy.

    Priority: dataset_value > default_value > global_value.
    A value of None means "not set" and should be skipped.

    Args:
        dataset_value: Per-dataset override (highest priority).
        default_value: Default section value (medium priority).
        global_value: Global fallback (lowest priority).

    Returns:
        The first non-None value, or None if all are None.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement resolve_field")


def build_eval_configs(
    datasets_raw: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    global_args: dict[str, Any] | None = None,
) -> list[EvalDatasetConfig]:
    """Build a list of EvalDatasetConfig from raw dicts with resolution.

    For each raw dataset dict, resolve these fields using the hierarchy
    (dataset -> defaults -> global_args):
        - temperature, top_p, max_response_len, n_samples,
          input_key, label_key

    Field name mappings for global_args (since global args may use different names):
        - "temperature" in global_args maps to temperature
        - "top_p" in global_args maps to top_p
        - "max_response_len" or "rollout_max_response_len" in global_args
        - "n_samples" or "n_samples_per_prompt" in global_args
        - "input_key" in global_args
        - "label_key" in global_args

    For max_response_len, try "max_response_len" first, then
    "rollout_max_response_len" in global_args.
    For n_samples, try "n_samples" first, then "n_samples_per_prompt".

    Args:
        datasets_raw: List of dicts, each with at least "name" and "path".
        defaults: Optional default overrides dict.
        global_args: Optional global argument dict.

    Returns:
        List of EvalDatasetConfig instances.

    Raises:
        ValueError: If any dataset dict is missing "name".
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement build_eval_configs")
