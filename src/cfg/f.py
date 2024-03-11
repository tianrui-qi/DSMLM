from src.cfg.config import ConfigTrainer

__all__ = ["f01"]


class f01(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.validset["scale"] = [4, 8, 8]
        self.model["feats"] = [1, 8, 16, 32, 64]
        self.runner["lr"] = 5e-4
