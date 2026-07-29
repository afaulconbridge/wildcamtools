"""Tests for the description-generation pipeline components."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from wildcamtools.lib.ai import Backend
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.pipeline import (
    ConcatenatingDescriptionReconciler,
    DescriptionImageBatchQuery,
    ExtractedBatch,
    ExtractedFrame,
    ExtractedFrames,
    ExtractedFramesWithResults,
    LlmDescriptionReconciler,
)
from wildcamtools.lib.ai.types import (
    BatchDescription,
    CombinedDescription,
)


class MockDescriptionLlm(AbstractLlm):
    """Mock LLM that returns BatchDescription or CombinedDescription as requested."""

    model: str
    backend: Backend
    url: str
    api_key: str | None = None
    call_count: int
    last_prompt: str
    last_images: list[Path]
    next_description: str
    next_combined_description: str
    should_fail: bool

    def __init__(
        self,
        model: str = "test-model",
        backend: Backend = Backend.OLLAMA,
        url: str = "http://test",
        next_description: str = "batch description",
        next_combined_description: str = "combined description",
        should_fail: bool = False,
    ) -> None:
        self.model = model
        self.backend = backend
        self.url = url
        self.next_description = next_description
        self.next_combined_description = next_combined_description
        self.should_fail = should_fail
        self.call_count = 0
        self.last_prompt = ""
        self.last_images = []

    def message_with_schema(
        self,
        message: str,
        images: Sequence[Path] = (),
        response_class: type = BatchDescription,
    ) -> Any:
        self.call_count += 1
        self.last_prompt = message
        self.last_images = list(images)
        if self.should_fail:
            raise RuntimeError("simulated LLM failure")
        if response_class is CombinedDescription:
            return CombinedDescription(
                description=self.next_combined_description,
                method="llm_combine",
                source_count=1,
            )
        if response_class is BatchDescription:
            return BatchDescription(description=self.next_description)
        raise ValueError(f"unexpected response class: {response_class}")


@pytest.fixture(name="sample_image_paths")
def fixture_sample_image_paths(tmp_path: Path) -> list[Path]:
    image_paths: list[Path] = []
    for i in range(3):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :] = [(i * 10) % 256, (i * 20) % 256, (i * 30) % 256]
        img_path = tmp_path / f"image_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        image_paths.append(img_path)
    return image_paths


class TestDescriptionImageBatchQuery:
    def test_default_prompt_used_when_unset(self, sample_image_paths: list[Path]) -> None:
        mock_llm = MockDescriptionLlm(next_description="hello world")
        query = DescriptionImageBatchQuery(llm=mock_llm)
        result = query.query_images(sample_image_paths)
        assert isinstance(result, BatchDescription)
        assert result.description == "hello world"
        assert mock_llm.call_count == 1

    def test_custom_prompt(self, sample_image_paths: list[Path]) -> None:
        mock_llm = MockDescriptionLlm(next_description="custom")
        query = DescriptionImageBatchQuery(llm=mock_llm, prompt="describe the scene")
        query.query_images(sample_image_paths)
        assert mock_llm.last_prompt == "describe the scene"

    def test_query_image_batches_uses_response_class(self, sample_image_paths: list[Path]) -> None:
        mock_llm = MockDescriptionLlm(next_description="desc 1")
        query = DescriptionImageBatchQuery(llm=mock_llm)
        batch_a = ExtractedBatch(
            selected_frames=[ExtractedFrame(path=p, frame_no=i) for i, p in enumerate(sample_image_paths)],
        )
        batch_b = ExtractedBatch(
            selected_frames=[ExtractedFrame(path=p, frame_no=i) for i, p in enumerate(sample_image_paths)],
        )

        enriched: ExtractedFramesWithResults[BatchDescription] = query.query_image_batches(
            ExtractedFrames(batches=[batch_a, batch_b]),
        )
        assert len(enriched.batches) == 2
        assert all(b.result is not None for b in enriched.batches)
        assert all(isinstance(b.result, BatchDescription) for b in enriched.batches)

    def test_empty_batch_raises(self) -> None:
        mock_llm = MockDescriptionLlm()
        query = DescriptionImageBatchQuery(llm=mock_llm)
        with pytest.raises(ValueError, match="Empty image batch"):
            query.query_images([])

    def test_sorts_images_before_querying(self, sample_image_paths: list[Path]) -> None:
        mock_llm = MockDescriptionLlm()
        query = DescriptionImageBatchQuery(llm=mock_llm)
        query.query_images(list(reversed(sample_image_paths)))
        assert mock_llm.last_images == sorted(sample_image_paths)


class TestConcatenatingDescriptionReconciler:
    def test_empty_input_returns_placeholder(self) -> None:
        reconciler = ConcatenatingDescriptionReconciler()
        result = reconciler.reconcile_results([])
        assert isinstance(result, BatchDescription)
        assert result.description == "No activity detected in this video."

    def test_single_input_returned_unchanged(self) -> None:
        reconciler = ConcatenatingDescriptionReconciler()
        single = BatchDescription(description="only one")
        result = reconciler.reconcile_results([single])
        assert result.description == "only one"

    def test_multiple_inputs_concatenated(self) -> None:
        reconciler = ConcatenatingDescriptionReconciler()
        a = BatchDescription(description="first segment.")
        b = BatchDescription(description="second segment.")
        c = BatchDescription(description="third segment.")
        result = reconciler.reconcile_results([a, b, c])
        assert result.description == "first segment.\n\nsecond segment.\n\nthird segment."

    def test_method_name(self) -> None:
        reconciler = ConcatenatingDescriptionReconciler()
        assert reconciler.method_name == "concatenate"


class TestLlmDescriptionReconciler:
    def test_empty_input_returns_placeholder_without_llm_call(self) -> None:
        mock_llm = MockDescriptionLlm()
        reconciler = LlmDescriptionReconciler(llm=mock_llm)
        result = reconciler.reconcile_results([])
        assert isinstance(result, BatchDescription)
        assert result.description == "No activity detected in this video."
        assert mock_llm.call_count == 0

    def test_single_input_returned_unchanged_without_llm_call(self) -> None:
        mock_llm = MockDescriptionLlm()
        reconciler = LlmDescriptionReconciler(llm=mock_llm)
        single = BatchDescription(description="only one")
        result = reconciler.reconcile_results([single])
        assert result.description == "only one"
        assert mock_llm.call_count == 0

    def test_multiple_inputs_combined_via_llm(self) -> None:
        mock_llm = MockDescriptionLlm(next_combined_description="merged description")
        reconciler = LlmDescriptionReconciler(llm=mock_llm)
        a = BatchDescription(description="first")
        b = BatchDescription(description="second")
        result = reconciler.reconcile_results([a, b])
        assert result.description == "merged description"
        assert mock_llm.call_count == 1
        assert "Segment 1" in mock_llm.last_prompt
        assert "Segment 2" in mock_llm.last_prompt
        assert "first" in mock_llm.last_prompt
        assert "second" in mock_llm.last_prompt

    def test_custom_prompt_used_with_descriptions_placeholder(self) -> None:
        mock_llm = MockDescriptionLlm(next_combined_description="x")
        reconciler = LlmDescriptionReconciler(
            llm=mock_llm,
            prompt="combine these: {descriptions}",
        )
        reconciler.reconcile_results([BatchDescription(description="a"), BatchDescription(description="b")])
        assert mock_llm.last_prompt == "combine these: [Segment 1]\na\n\n[Segment 2]\nb"

    def test_llm_failure_falls_back_to_concatenation(self) -> None:
        mock_llm = MockDescriptionLlm(should_fail=True)
        reconciler = LlmDescriptionReconciler(llm=mock_llm)
        a = BatchDescription(description="alpha")
        b = BatchDescription(description="beta")
        result = reconciler.reconcile_results([a, b])
        assert result.description == "alpha\n\nbeta"

    def test_custom_fallback_reconciler(self) -> None:
        mock_llm = MockDescriptionLlm(should_fail=True)
        # Custom fallback returns the first input only.

        class FirstOnlyReconciler(ConcatenatingDescriptionReconciler):
            def reconcile_results(self, results: Iterable[BatchDescription]) -> BatchDescription:
                first = next(iter(results))
                return BatchDescription(description=first.description)

        reconciler = LlmDescriptionReconciler(llm=mock_llm, fallback=FirstOnlyReconciler())  # type: ignore[arg-type]
        a = BatchDescription(description="alpha")
        b = BatchDescription(description="beta")
        result = reconciler.reconcile_results([a, b])
        assert result.description == "alpha"

    def test_method_name(self) -> None:
        reconciler = LlmDescriptionReconciler(llm=MockDescriptionLlm())
        assert reconciler.method_name == "llm_combine"

    def test_last_method_name_records_fallback(self) -> None:
        mock_llm = MockDescriptionLlm(should_fail=True)
        reconciler = LlmDescriptionReconciler(llm=mock_llm)
        reconciler.reconcile_results([BatchDescription(description="a"), BatchDescription(description="b")])
        assert reconciler.last_method_name == "concatenate"

    def test_last_method_name_records_llm_combine_on_success(self) -> None:
        mock_llm = MockDescriptionLlm(next_combined_description="merged")
        reconciler = LlmDescriptionReconciler(llm=mock_llm)
        reconciler.reconcile_results([BatchDescription(description="a"), BatchDescription(description="b")])
        assert reconciler.last_method_name == "llm_combine"
