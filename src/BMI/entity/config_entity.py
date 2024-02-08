from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_augmentation: bool
    params_image_size: List[int]
    params_batch_size: int
    params_include_top: bool
    params_epochs: int
    params_output_variables: List[dict]
    params_weights: str
    params_learning_rate: float
