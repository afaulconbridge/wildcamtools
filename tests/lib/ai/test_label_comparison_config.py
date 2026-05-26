import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wildcamtools.lib.ai import LlmConfig
from wildcamtools.lib.ai.label_comparison import ExactLabelComparator, LLMLabelComparator
from wildcamtools.lib.ai.label_comparison_config import LabelComparisonConfig, LabelComparisonType


class TestLabelComparisonConfig:
    def test_default_values(self) -> None:
        config = LabelComparisonConfig()
        assert config.comparator_type == LabelComparisonType.EXACT
        assert config.llm is None
        assert config.prompt is None
        assert config.cache_enabled is True

    def test_custom_values(self) -> None:
        config = LabelComparisonConfig(
            comparator_type=LabelComparisonType.LLM,
            llm=LlmConfig(model="test-model"),
            prompt="Custom prompt",
            cache_enabled=False,
        )
        assert config.comparator_type == LabelComparisonType.LLM
        assert config.llm is not None
        assert config.llm.model == "test-model"
        assert config.prompt == "Custom prompt"
        assert config.cache_enabled is False

    def test_create_exact_comparator(self) -> None:
        config = LabelComparisonConfig()
        comparator = config.create_comparator()
        assert isinstance(comparator, ExactLabelComparator)
        assert comparator.method_name == "exact"

    def test_create_llm_comparator(self) -> None:
        config = LabelComparisonConfig(
            comparator_type=LabelComparisonType.LLM,
            llm=LlmConfig(model="test-model"),
        )
        comparator = config.create_comparator()
        assert isinstance(comparator, LLMLabelComparator)
        assert comparator.method_name == "llm"

    def test_create_llm_comparator_without_llm_config_raises_error(self) -> None:
        config = LabelComparisonConfig(comparator_type=LabelComparisonType.LLM)
        with pytest.raises(ValueError, match="LLM configuration is required"):
            config.create_comparator()

    def test_create_comparator_with_custom_prompt(self) -> None:
        config = LabelComparisonConfig(
            comparator_type=LabelComparisonType.LLM,
            llm=LlmConfig(model="test-model"),
            prompt="Custom: {result} vs {label}",
        )
        comparator = config.create_comparator()
        assert isinstance(comparator, LLMLabelComparator)
        assert comparator.prompt == "Custom: {result} vs {label}"

    def test_create_comparator_with_cache_disabled(self) -> None:
        config = LabelComparisonConfig(
            comparator_type=LabelComparisonType.LLM,
            llm=LlmConfig(model="test-model"),
            cache_enabled=False,
        )
        comparator = config.create_comparator()
        assert isinstance(comparator, LLMLabelComparator)
        assert comparator._cache is None

    def test_from_json(self, tmp_path: Path) -> None:
        json_content = """
        {
            "comparator_type": "llm",
            "llm": {
                "model": "test-model",
                "backend": "ollama",
                "url": "http://localhost:8080/v1"
            },
            "prompt": "Custom prompt",
            "cache_enabled": false
        }
        """
        config_file = tmp_path / "config.json"
        config_file.write_text(json_content)

        config = LabelComparisonConfig.from_json(config_file)
        assert config.comparator_type == LabelComparisonType.LLM
        assert config.llm is not None
        assert config.llm.model == "test-model"
        assert config.prompt == "Custom prompt"
        assert config.cache_enabled is False

    def test_to_json(self, tmp_path: Path) -> None:
        config = LabelComparisonConfig(
            comparator_type=LabelComparisonType.LLM,
            llm=LlmConfig(model="test-model"),
            prompt="Custom prompt",
        )
        output_file = tmp_path / "output_config.json"

        config.to_json(output_file)

        assert output_file.exists()
        loaded_data = json.loads(output_file.read_text())
        assert loaded_data["comparator_type"] == "llm"
        assert loaded_data["llm"]["model"] == "test-model"
        assert loaded_data["prompt"] == "Custom prompt"

    def test_roundtrip_json(self, tmp_path: Path) -> None:
        original = LabelComparisonConfig(
            comparator_type=LabelComparisonType.LLM,
            llm=LlmConfig(model="qwen3.5:cloud"),
            prompt="Test prompt",
            cache_enabled=False,
        )

        output_file = tmp_path / "roundtrip.json"
        original.to_json(output_file)
        restored = LabelComparisonConfig.from_json(output_file)

        assert restored.comparator_type == original.comparator_type
        assert restored.llm is not None
        assert restored.llm.model == original.llm.model
        assert restored.prompt == original.prompt
        assert restored.cache_enabled == original.cache_enabled

    def test_invalid_comparator_type(self) -> None:
        with pytest.raises(ValidationError):
            LabelComparisonConfig.model_validate({"comparator_type": "invalid"})

    def test_empty_prompt(self) -> None:
        with pytest.raises(ValidationError):
            LabelComparisonConfig(prompt="")

    def test_serialization(self) -> None:
        config = LabelComparisonConfig()
        data = config.model_dump()
        assert data["comparator_type"] == "exact"
        assert data["llm"] is None
        assert data["prompt"] is None
        assert data["cache_enabled"] is True

    def test_from_dict_with_enum_string(self) -> None:
        config = LabelComparisonConfig.model_validate({"comparator_type": "exact"})
        assert config.comparator_type == LabelComparisonType.EXACT

    def test_from_dict_with_llm_config(self) -> None:
        config = LabelComparisonConfig.model_validate({
            "comparator_type": "llm",
            "llm": {"model": "test-model", "backend": "ollama"},
        })
        assert config.comparator_type == LabelComparisonType.LLM
        assert config.llm is not None
        assert config.llm.model == "test-model"


class TestIntegration:
    def test_full_config_with_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("TEST_API_KEY", "secret-key-123")

        json_content = """
        {
            "comparator_type": "llm",
            "llm": {
                "model": "test-model",
                "backend": "ollama",
                "url": "http://localhost:8080/v1",
                "api_key": "${TEST_API_KEY}"
            },
            "prompt": "Test with env var",
            "cache_enabled": true
        }
        """
        config_file = tmp_path / "config_with_env.json"
        config_file.write_text(json_content)

        config = LabelComparisonConfig.from_json(config_file)
        assert config.llm is not None
        assert config.llm.api_key == "secret-key-123"
