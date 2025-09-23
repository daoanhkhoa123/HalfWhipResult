import lightning as lit
from lightning.pytorch.cli import LightningCLI

from modelm.whisper import WhipserMixed
from datam.litdataloader import VSAVDataModule

import logging
from datetime import datetime
import os

def setup_logger(run_name:str = "Sample Run"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir = os.path.join(base_dir, "logs")

    timestamp = datetime.now()
    log_file = os.path.join(log_dir, f"train_{timestamp}_{run_name}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_file

def main():
    LightningCLI(
        WhipserMixed,
        VSAVDataModule,
    )

if __name__ == "__main__":
    main()