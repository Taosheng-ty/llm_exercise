"""Tests for Exercise 06: Eval Config Builder."""

import importlib.util
import os

import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex06", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EvalDatasetConfig = _mod.EvalDatasetConfig
resolve_field = _mod.resolve_field
build_eval_configs = _mod.build_eval_configs


class TestEvalDatasetConfig:
    def test_default_metadata_overrides(self):
        cfg = EvalDatasetConfig(name="test", path="/data")
        assert cfg.metadata_overrides == {}

    def test_invalid_metadata_overrides(self):
        with pytest.raises(TypeError):
            EvalDatasetConfig(name="test", path="/data", metadata_overrides="bad")

    def test_none_metadata_overrides_becomes_dict(self):
        cfg = EvalDatasetConfig(name="test", path="/data", metadata_overrides=None)
        assert cfg.metadata_overrides == {}

    def test_inject_metadata_basic(self):
        cfg = EvalDatasetConfig(
            name="test", path="/data",
            rm_type="gpqa",
            metadata_overrides={"key1": "val1"},
        )
        result = cfg.inject_metadata({"existing": "data"})
        assert result == {"existing": "data", "rm_type": "gpqa", "key1": "val1"}

    def test_inject_metadata_none_input(self):
        cfg = EvalDatasetConfig(name="test", path="/data", rm_type="math")
        result = cfg.inject_metadata(None)
        assert result == {"rm_type": "math"}

    def test_inject_metadata_non_dict_input(self):
        cfg = EvalDatasetConfig(name="test", path="/data")
        result = cfg.inject_metadata("not a dict")
        assert result == {}

    def test_inject_metadata_override_existing(self):
        cfg = EvalDatasetConfig(
            name="test", path="/data",
            metadata_overrides={"key": "new_value"},
        )
        result = cfg.inject_metadata({"key": "old_value"})
        assert result["key"] == "new_value"


class TestResolveField:
    def test_dataset_wins(self):
        assert resolve_field(0.7, 0.5, 0.3) == 0.7

    def test_default_when_dataset_none(self):
        assert resolve_field(None, 0.5, 0.3) == 0.5

    def test_global_when_others_none(self):
        assert resolve_field(None, None, 0.3) == 0.3

    def test_all_none(self):
        assert resolve_field(None, None, None) is None

    def test_zero_is_valid(self):
        assert resolve_field(0, 1, 2) == 0

    def test_empty_string_is_valid(self):
        assert resolve_field("", "default", "global") == ""


class TestBuildEvalConfigs:
    def test_basic_build(self):
        raw = [{"name": "gpqa", "path": "/data/gpqa.jsonl"}]
        configs = build_eval_configs(raw)
        assert len(configs) == 1
        assert configs[0].name == "gpqa"
        assert configs[0].path == "/data/gpqa.jsonl"

    def test_missing_name_raises(self):
        with pytest.raises(ValueError):
            build_eval_configs([{"path": "/data/test.jsonl"}])

    def test_dataset_overrides_default(self):
        raw = [{"name": "test", "path": "/data", "temperature": 0.7}]
        defaults = {"temperature": 0.5}
        configs = build_eval_configs(raw, defaults=defaults)
        assert configs[0].temperature == 0.7

    def test_default_used_when_dataset_missing(self):
        raw = [{"name": "test", "path": "/data"}]
        defaults = {"temperature": 0.5}
        configs = build_eval_configs(raw, defaults=defaults)
        assert configs[0].temperature == 0.5

    def test_global_fallback(self):
        raw = [{"name": "test", "path": "/data"}]
        global_args = {"temperature": 0.3}
        configs = build_eval_configs(raw, global_args=global_args)
        assert configs[0].temperature == 0.3

    def test_hierarchy_dataset_default_global(self):
        raw = [{"name": "test", "path": "/data", "temperature": 0.9}]
        defaults = {"temperature": 0.5}
        global_args = {"temperature": 0.1}
        configs = build_eval_configs(raw, defaults=defaults, global_args=global_args)
        assert configs[0].temperature == 0.9

    def test_global_alias_max_response_len(self):
        raw = [{"name": "test", "path": "/data"}]
        global_args = {"rollout_max_response_len": 2048}
        configs = build_eval_configs(raw, global_args=global_args)
        assert configs[0].max_response_len == 2048

    def test_global_alias_n_samples(self):
        raw = [{"name": "test", "path": "/data"}]
        global_args = {"n_samples_per_prompt": 16}
        configs = build_eval_configs(raw, global_args=global_args)
        assert configs[0].n_samples == 16

    def test_multiple_datasets(self):
        raw = [
            {"name": "gpqa", "path": "/gpqa", "rm_type": "gpqa"},
            {"name": "math", "path": "/math", "rm_type": "math", "temperature": 0.0},
        ]
        defaults = {"temperature": 0.7, "max_response_len": 1024}
        configs = build_eval_configs(raw, defaults=defaults)
        assert len(configs) == 2
        assert configs[0].temperature == 0.7
        assert configs[1].temperature == 0.0
        assert configs[0].max_response_len == 1024
        assert configs[1].max_response_len == 1024

    def test_metadata_overrides_passed_through(self):
        raw = [{"name": "test", "path": "/data", "metadata_overrides": {"custom": True}}]
        configs = build_eval_configs(raw)
        assert configs[0].metadata_overrides == {"custom": True}
