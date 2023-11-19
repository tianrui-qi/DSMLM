import torch.backends.cudnn

import sml.config
from sml.config import e21 as config
import sml.runner


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    if isinstance(config(), sml.config.TrainerConfig): 
        sml.runner.Trainer(config()).fit()
    if isinstance(config(), sml.config.EvaluerConfig): 
        sml.runner.Evaluer(config()).fit()
