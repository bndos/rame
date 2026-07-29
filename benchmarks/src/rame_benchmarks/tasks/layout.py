from __future__ import annotations

from pathlib import Path

from datasets import Image, load_dataset

from rame_benchmarks.tasks._images import write_image
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

    def load_data(self, output_dir: Path, **kwargs: object) -> None:
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
        self.images = [write_image(record, output_dir) for record in ds]
        self.data_loaded = True


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
