from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Image, load_dataset

from rame_benchmarks.samples import ImageSample
from rame_benchmarks.tasks.abstask import (
    AbsTask,
    DatasetMetadata,
    TaskMetadata,
    TaskName,
)


class LayoutTaskBase(AbsTask):
    metadata = TaskMetadata(
        name=TaskName.LAYOUT,
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


class LayoutTask(LayoutTaskBase):
    pass


class LayoutMicroTask(LayoutTaskBase):
    metadata = TaskMetadata(
        name=TaskName.LAYOUT_MICRO,
        dataset=DatasetMetadata(
            path="creative-graphic-design/PubLayNet",
            split="test",
        ),
    )
    _sample_range = range(128)
