from logging import getLogger


logger = getLogger(__name__)

class PreTrainLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
