import torch.backends.cudnn

import sml.config
import sml.runner


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    for config in (
        sml.config.e10_4(), sml.config.e10_8(),
        sml.config.e11_4(), sml.config.e11_8(),
        sml.config.e13_8(),
    ):
        if isinstance(config, sml.config.TrainerConfig): 
            sml.runner.Trainer(config).fit()
        if isinstance(config, sml.config.EvaluerConfig): 
            sml.runner.Evaluer(config).fit()
