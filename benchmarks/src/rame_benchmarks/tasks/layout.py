from __future__ import annotations

from pathlib import Path

from datasets import Image, load_dataset

from rame_benchmarks.tasks._images import write_image
from rame_benchmarks.tasks.abstask import AbsTask, TaskName


class LayoutTask(AbsTask):
    name = TaskName.LAYOUT

    _repo_id = "creative-graphic-design/PubLayNet"
    _split = "test"

    images: list[Path]

    def load_data(
        self,
        output_dir: Path,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> None:
        if self.data_loaded:
            return

        ds = load_dataset(self._repo_id, split=self._split)
        ds = ds.select_columns(["file_name", "image"]).cast_column(
            "image", Image(decode=False)
        )

        start = offset
        end = len(ds) if limit is None else min(offset + limit, len(ds))
        ds = ds.select(range(start, end))

        output_dir.mkdir(parents=True, exist_ok=True)
        self.images = [write_image(record, output_dir) for record in ds]
        self.data_loaded = True
