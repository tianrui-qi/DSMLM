import config
from sml import Trainer, Evaluator


def sml_train():
    for cfg in config.getConfig():
        cfg.train()
        trainer = Trainer(cfg)
        trainer.train()


def sml_eval():
    for cfg in config.getConfig():
        cfg.eval()
        evaluator = Evaluator(cfg)
        evaluator.eval()


if __name__ == "__main__":
    sml_eval()
