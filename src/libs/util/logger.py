from pathlib import Path
from logging import getLogger
from typing import Dict, Any

import pandas as pd

logger = getLogger(__name__)


class TrainLogger(object):
    def __init__(self, log_dir: Path, resume: bool=False, epoch: int=0) -> None:
        self.log_dir = log_dir
        self.loss_dir = log_dir / "loss"
        self.loss_dir.mkdir(exist_ok=True)
        self.log_path = self.loss_dir / f"{epoch}.tsv"
        self.epoch = epoch
        self.columns = [
            "total",
            "GAN loss/G", "GAN loss/D",
            "prosody loss/total", "prosody loss without sign/total",
            "Regularization/Energy mean", "Regularization/Pitch mean",
            "Regularization/Energy intonation", "Regularization/Pitch intonation",
        ]

        if resume:
            self.df = self._load_log()
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def _load_log(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.log_path, sep="\t")
            return df
        except FileNotFoundError as err:
            logger.exception(f"{err}")
            raise err

    def _save_log(self) -> None:
        self.df.to_csv(self.log_path, index=False, sep="\t")

    def update(
        self,
        loss_log: Dict[str, Any]
    ) -> None:
        tmp = pd.DataFrame(
            [{key: f"{loss_log[key]:.5f}" for key in self.columns}],
            columns=self.columns,
        )

        if self.epoch != loss_log["epoch"]:
            self.epoch = loss_log["epoch"]
            self.log_path = self.loss_dir / f"{self.epoch}.tsv"
            self.df = pd.DataFrame(columns=self.columns)
        self.df = pd.concat([self.df, tmp], ignore_index=True)
        self._save_log()
