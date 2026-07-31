from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from huggingface_hub import hf_hub_download

from rame_benchmarks.models.models_protocols import BenchmarkModel
from rame_benchmarks.samples import ImageSample
from rame_benchmarks.tasks.abstask import (
    AbsTask,
    DatasetMetadata,
    TaskMetadata,
    TaskMetric,
    TaskName,
    TaskResult,
)
from rame_benchmarks.utils import chunked


class LayoutTaskBase(AbsTask):
    metadata = TaskMetadata(
        name=TaskName.LAYOUT_THROUGHPUT,
        dataset=DatasetMetadata(
            path="opendatalab/OmniDocBench",
            split="main",
        ),
    )
    _sample_count: int | None = None

    def __init__(self) -> None:
        super().__init__()
        self.image_paths: list[Path] = []
        self.image_samples: list[ImageSample] = []

    def load_data(self, output_dir: Path, **_kwargs: Any) -> None:
        if self.data_loaded:
            return

        annotations_path = Path(
            hf_hub_download(
                self.metadata.dataset.path,
                "OmniDocBench.json",
                repo_type="dataset",
                revision=self.metadata.dataset.revision,
            )
        )
        records = json.loads(annotations_path.read_text(encoding="utf-8"))
        if self._sample_count is not None:
            records = records[: self._sample_count]

        output_dir.mkdir(parents=True, exist_ok=True)
        self.image_samples = [
            self.load_image_sample(record, output_dir) for record in records
        ]
        self.image_paths = [sample.path for sample in self.image_samples]
        self.data_loaded = True

    def load_image_sample(
        self, record: dict[str, Any], output_dir: Path
    ) -> ImageSample:
        image_name = record["page_info"]["image_path"]
        source_path = Path(
            hf_hub_download(
                self.metadata.dataset.path,
                f"images/{image_name}",
                repo_type="dataset",
                revision=self.metadata.dataset.revision,
            )
        )
        image_bytes = source_path.read_bytes()
        image_path = ImageSample.cache_path(image_name, output_dir)
        sample = ImageSample.from_bytes(image_path, image_bytes)
        sample.write_original_bytes(image_bytes)
        return sample

    def _evaluate(
        self,
        model: BenchmarkModel,
        *,
        batch_size: int,
        warmup: int,
        repeats: int,
    ) -> TaskResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if warmup < 0:
            raise ValueError("warmup must be greater than or equal to zero")
        if repeats <= 0:
            raise ValueError("repeats must be greater than zero")
        if not self.data_loaded:
            raise RuntimeError("task data must be loaded before evaluation")

        batches = list(chunked(self.image_samples, batch_size))
        for _ in range(warmup):
            for batch in batches:
                model.detect_layout_many(batch, batch_size=batch_size)

        started = perf_counter()
        for _ in range(repeats):
            for batch in batches:
                model.detect_layout_many(batch, batch_size=batch_size)
        elapsed_s = perf_counter() - started
        total_samples = len(self.image_samples) * repeats

        return TaskResult(
            task_name=self.name,
            metrics=(
                TaskMetric("samples", len(self.image_samples)),
                TaskMetric("warmup", warmup),
                TaskMetric("repeats", repeats),
                TaskMetric("total_samples", total_samples),
                TaskMetric("batches", len(batches) * repeats),
                TaskMetric("elapsed", elapsed_s, "s"),
                TaskMetric(
                    "throughput",
                    total_samples / elapsed_s,
                    "samples/s",
                ),
            ),
        )


class LayoutThroughputTask(LayoutTaskBase):
    pass


class LayoutThroughputMicroTask(LayoutTaskBase):
    metadata = TaskMetadata(
        name=TaskName.LAYOUT_THROUGHPUT_MICRO,
        dataset=DatasetMetadata(
            path="opendatalab/OmniDocBench",
            split="main",
        ),
    )
    _sample_count = 128
