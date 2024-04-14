import torch
import torch.cuda
import torch.backends.cudnn

import numpy as np

import os
import random
import argparse
import inspect
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cupy')

import src


__all__ = []


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def main() -> None:
    setSeed(42)
    args = getArgs()
    if args.mode == "evalu":
        config = src.ConfigEvaluer(**vars(args))
        src.Evaluer(**config.runner,
            evaluset=src.RawDataset(**config.evaluset), 
        ).fit()
    if args.mode == "train":
        config = getattr(src, args.config)()
        src.Trainer(**config.runner,
            trainset=src.SimDataset(**config.trainset), 
            validset=src.RawDataset(**config.validset), 
            model=src.ResAttUNet(**config.model), 
        ).fit()


def setSeed(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    np.random.seed(seed)
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  # python hash seed


def getArgs():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest='mode', help='Modes of operation: train or evalu.'
    )

    # train mode
    parser_train = subparsers.add_parser(
        'train', 
        help="Training mode. " + 
        "Type `python main.py train -h` for more information."
    )
    parser_train.add_argument(
        "-config", type=str, required=True, dest="config",
        choices=[
            name for name, obj in inspect.getmembers(src) 
            if inspect.isclass(obj) and issubclass(obj, src.ConfigTrainer) and 
            obj is not src.ConfigTrainer
        ],
        help="Choose a config class for training that inherit from " +
        "src.ConfigTrainer. Choices list all match object in src." + 
        "Config class is used instead of argparse since parameters may " + 
        "frequently add, remove, or change during " + 
        "model development and testing."
    )

    # evalu mode
    parser_evalu = subparsers.add_parser(
        'evalu',
        help="Evaluation mode. " + 
        "Type `python main.py evalu -h` for more information."
    )
    parser_evalu.add_argument(
        "-s", type=int, required=False, dest="scale", 
        choices=[4, 8], default=4,
        help="Scale up factor, 4 or 8. Default: 4."
    )
    parser_evalu.add_argument(
        "-r", type=int, required=False, dest="rng_sub_user",
        default=None, nargs="+",
        help="Range of the sub-region of the frames to predict. " + 
        "Due to limited memory, we cut whole frames into patches, " + 
        "i.e., sub-regions and predict them separately. " + 
        "Please type six int separated by space as the subframe " + 
        "start (inclusive) and end (exclusive) index for each dimension, " +
        "i.e., `-r 0 1 8 12 9 13`. " + 
        "If you not sure about the number of subframe for each dimension " +
        "you can select, do not specify this parameter; " + 
        "the code will print the range you can select and " + 
        "ask you to type the range. Default: None."
    )
    parser_evalu.add_argument(
        "-L", type=str, required=True, dest="frames_load_fold",
        help="Path to the frames load folder. Note that the code will " + 
        "predict all the frames under this folder. " + 
        "Thus, if you want to predict portion of the frames, " + 
        "please copy them to a new folder and specify this parameter " + 
        "with that new folder."
    )
    parser_evalu.add_argument(
        "-S", type=str, required=False, dest="data_save_fold",
        default=None,
        help="Path to the data save folder. No need to specify " +
        "when stride or window is set as non-zero. Default: None."
    )
    parser_evalu.add_argument(
        "-C", type=str, required=False, dest="ckpt_load_path",
        default=None,
        help="Path to the checkpoint load file. Default: `ckpt/e08/340.ckpt` " + 
        "or `ckpt/e10/450.ckpt` when scale up factor is 4 or 8."
    )
    parser_evalu.add_argument(
        "-T", type=str, required=False, dest="temp_save_fold", 
        default=None,
        help="Path to the temporary save folder for drifting analysis. " + 
        "Must be specified when drift correction will be performed. " +
        "Recomend to specify different path for different dataset. " +
        "Default: `os.path.dirname(FRAMES_LOAD_FOLD)/temp/`."
    )
    parser_evalu.add_argument(
        "-stride", type=int, required=False, dest="stride", 
        default=0,
        help="Step size of the drift corrector, unit frames. " + 
        "Window size must larger or equal to stride and divisible by " + 
        "stride. Should set with window at the same time. Default: 0."
    )
    parser_evalu.add_argument(
        "-window", type=int, required=False, dest="window", 
        default=0,
        help="Number of frames in each window, unit frames. " + 
        "Window size must larger or equal to stride and divisible by " + 
        "stride. Should set with stride at the same time. Default: 0."
    )
    parser_evalu.add_argument(
        "-m", type=str, required=False, dest="method",
        choices=["DCC", "MCC", "RCC"], default=None,
        help="Drift correction method, DCC, MCC, or RCC. " + 
        "Must be set if you want to evaluate with drift correction. " +
        "DCC run very fast where MCC and RCC is more accurate. " + 
        "We suggest to use DCC to test the window size first and " + 
        "then use MCC or RCC to calculate the final drift. Default: None."
    )
    parser_evalu.add_argument(
        "-b", type=int, required=False, dest="batch_size",
        default=1,
        help="Batch size. Set this value according to your GPU memory. " +
        "Note that the product of rng_sub_user must divisible " + 
        "by batch_size. Default: 1."
    )
    parser_evalu.add_argument(
        "-w", type=int, required=False, dest="num_workers",
        default=1,
        help="Number of workers for dataloader. Set this value according " + 
        "to your CPU. Default: 1."
    )

    args = parser.parse_args()

    # set default value
    if args.mode == 'evalu':
        if args.ckpt_load_path is None:
            if args.scale == 4: args.ckpt_load_path = "ckpt/e08/340.ckpt"
            if args.scale == 8: args.ckpt_load_path = "ckpt/e10/450.ckpt"
        if args.temp_save_fold is None:
            args.temp_save_fold = os.path.join(
                os.path.dirname(os.path.normpath(args.frames_load_fold)), "temp"
            )

    return args


if __name__ == "__main__": main()
