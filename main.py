import config

from sml_dl import Trainer, Evaluator


def sml_dl_train():
    for cfg in config.getConfig():
        cfg.train()
        trainer = Trainer(cfg)
        trainer.train()


def sml_dl_eval():
    for cfg in config.getConfig():
        cfg.eval()
        evaluator = Evaluator(cfg)
        evaluator.eval()


if __name__ == "__main__":
    sml_dl_eval()
