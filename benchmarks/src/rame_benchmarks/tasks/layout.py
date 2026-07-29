from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from datasets import Image, load_dataset

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
            path="creative-graphic-design/PubLayNet",
            split="test",
        ),
    )
    _sample_range: range | None = None

    def __init__(self) -> None:
        super().__init__()
        self.image_paths: list[Path] = []
        self.image_samples: list[ImageSample] = []

    def load_data(self, output_dir: Path, **_kwargs: Any) -> None:
        if self.data_loaded:
            return

        ds = load_dataset(
            self.metadata.dataset.path,
            split=self.metadata.dataset.split,
            revision=self.metadata.dataset.revision,
        )
        ds = ds.select_columns(["file_name", "image"]).cast_column(
            "image", Image(decode=False)
        )

        if self._sample_range is not None:
            ds = ds.select(self._sample_range)

        output_dir.mkdir(parents=True, exist_ok=True)
        self.image_samples = [
            self.load_image_sample(record, output_dir) for record in ds
        ]
        self.image_paths = [sample.path for sample in self.image_samples]
        self.data_loaded = True

    def load_image_sample(
        self, record: dict[str, Any], output_dir: Path
    ) -> ImageSample:
        image_bytes = record["image"]["bytes"]
        image_path = ImageSample.cache_path(record["file_name"], output_dir)
        sample = ImageSample.from_bytes(image_path, image_bytes)
        sample.write_original_bytes(image_bytes)
        return sample

    def _evaluate(self, model: BenchmarkModel, *, batch_size: int) -> TaskResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not self.data_loaded:
            raise RuntimeError("task data must be loaded before evaluation")

        batches = list(chunked(self.image_samples, batch_size))
        started = perf_counter()
        for batch in batches:
            model.detect_layout_many(batch, batch_size=batch_size)
        elapsed_s = perf_counter() - started

        return TaskResult(
            task_name=self.name,
            metrics=(
                TaskMetric("samples", len(self.image_samples)),
                TaskMetric("batches", len(batches)),
                TaskMetric("elapsed", elapsed_s, "s"),
                TaskMetric(
                    "throughput",
                    len(self.image_samples) / elapsed_s,
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
            path="creative-graphic-design/PubLayNet",
            split="test",
        ),
    )
    _sample_range = range(128)
