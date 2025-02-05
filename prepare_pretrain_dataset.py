# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import time
import click
import torch
from pathlib import Path
from typing import Literal
from litgpt.tokenizer import Tokenizer
from litdata.processing.data_processor import DataChunkRecipe, DataProcessor
from litgpt.utils import extend_checkpoint_dir


class MyDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int,
        text_key: str,
        recursive: bool = True,
        file_type: Literal["jsonl", "csv", "parquet"] = "jsonl",
    ):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer
        self.recursive = recursive
        self.text_key = text_key
        self.file_type = file_type

    def prepare_structure(self, input_dir):
        files = []
        for root, _, filenames in os.walk(input_dir, followlinks=True):
            for filename in filenames:
                if filename.endswith(f".{self.file_type}"):
                    files.append(os.path.join(root, filename))
        return (
            files
            if self.recursive
            else [f for f in files if os.path.dirname(f) == input_dir]
        )

    def prepare_item(self, filepath):
        # 这里之所以使用.to(torch.int16)，是因为我使用的tokenizer词表大小为6400，
        # 而litgpt默认使用torch.int32，所以转换为torch.int16可以节省一半的磁盘储存
        if self.file_type == "jsonl":
            import json

            with open(filepath, "rb") as f:
                for line in f.readlines():
                    js = json.loads(line)
                    text = js[self.text_key]
                    yield self.tokenizer.encode(text, eos=True).to(torch.int16)
        elif self.file_type == "csv":
            import pandas as pd

            df = pd.read_csv(filepath)
            for text in df[self.text_key]:
                yield self.tokenizer.encode(text, eos=True).to(torch.int16)
        elif self.file_type == "parquet":
            import pandas as pd

            df = pd.read_parquet(filepath)
            for text in df[self.text_key]:
                yield self.tokenizer.encode(text, eos=True).to(torch.int16)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")


@click.command()
@click.option("--input_dir", type=click.Path(exists=True), required=True)
@click.option("--output_dir", type=click.Path(), required=True)
@click.option("--tokenizer_path", type=click.Path(exists=True), required=True)
@click.option(
    "--text_key",
    type=str,
    required=True,
    help="The key of the text column in the dataset",
)
@click.option(
    "--file_type", type=click.Choice(["jsonl", "csv", "parquet"]), default="jsonl"
)
@click.option(
    "--fast_dev_run",
    is_flag=True,
    default=False,
    help="Run the data processor in fast dev mode(used for debugging)",
)
def prepare(
    input_dir: str,
    output_dir: str,
    tokenizer_path: str,
    text_key: str,
    file_type: Literal["jsonl", "csv", "parquet"] = "jsonl",
    chunk_size: int = (2049 * 8012),
    fast_dev_run: bool = False,
) -> None:
    # 由于litdata的缓存机制，处理过程中会将原始数据缓存后做处理，最后会删除缓存数据
    # 默认会使用/tmp目录，这里为了防止/tmp目录空间不足，使用input_dir的上一级目录作为缓存目录
    os.environ["DATA_OPTIMIZER_DATA_CACHE_FOLDER"] = f"{input_dir}/../cache"
    os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = f"{input_dir}/../chunks"
    tokenizer_path = extend_checkpoint_dir(Path(tokenizer_path))
    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = MyDataRecipe(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        text_key=text_key,
        file_type=file_type,
    )
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    prepare()
