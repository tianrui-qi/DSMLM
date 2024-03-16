from src.cfg.config import ConfigTrainer


__all__ = ["f01"]


class f01(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["psf_load_path"] = "data/psf.tif"

        #self.runner["max_epoch"] = 450
        #self.runner["ckpt_load_path"] = "ckpt/e08/340.ckpt"
        #self.runner["lr"] = 1e-6

        #self.runner["max_epoch"] = 480
        #self.runner["ckpt_load_path"] = "ckpt/f01/450.ckpt"
        #self.runner["lr"] = 1e-7

        self.runner["max_epoch"] = 565
        self.runner["ckpt_load_path"] = "ckpt/f01/480.ckpt"
        self.runner["lr"] = 5e-6
