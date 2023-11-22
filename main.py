import torch.backends.cudnn

import sml.config
import sml.runner


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    for config in (
        sml.config.e08_4(), sml.config.e09_4(), 
        sml.config.e12_8(), sml.config.e21_8(),
    ):
        if isinstance(config, sml.config.TrainerConfig): 
            sml.runner.Trainer(config).fit()
        if isinstance(config, sml.config.EvaluerConfig): 
            sml.runner.Evaluer(config).fit()
