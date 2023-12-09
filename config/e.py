from .config import *

__all__ = [
    "e08", "e08_4", "e08_8", 
    "e09", "e09_8", "e10", "e10_8", "e11", "e11_8",
]


""" scale up by 8
features number : [1, 16, 32, 64, 128, 256, 512]
Trainable paras : 21,401,393
Training   speed: 1.10 steps /s ( 10 iterations/step)
Validation speed: 1.74 steps /s ( 10 iterations/step)
Evaluation speed: 2.86 frames/s ( 16 subframes/frame)
"""


class e11_8(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        self.RawDataset["scale"] = [4, 8, 8]
        self.ResAttUNet["feats"] = [1, 16, 32, 64, 128, 256, 512]
        self.Evaluer["ckpt_load_path"] = self.ckpt_disk + "e11/340"


class e11(Config):
    def __init__(self) -> None:
        super().__init__("train")
        self.SimDataset["scale_list"] = [4, 8, 16]
        self.RawDataset["scale"] = [4, 8, 8]
        self.ResAttUNet["feats"] = [1, 16, 32, 64, 128, 256, 512]
        self.Trainer["max_epoch"] = 340
        self.Trainer["ckpt_load_path"] = self.ckpt_disk + "e08/200"
        self.Trainer["lr"] = 1e-6


class e10_8(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        self.RawDataset["scale"] = [4, 8, 8]
        self.ResAttUNet["feats"] = [1, 16, 32, 64, 128, 256, 512]
        self.Evaluer["ckpt_load_path"] = self.ckpt_disk + "e10/340"


class e10(Config):
    def __init__(self) -> None:
        super().__init__("train")
        self.SimDataset["scale_list"] = [8, 16]
        self.RawDataset["scale"] = [4, 8, 8]
        self.ResAttUNet["feats"] = [1, 16, 32, 64, 128, 256, 512]

        """
        self.Trainer["max_epoch"] = 340
        self.Trainer["ckpt_load_path"] = self.ckpt_disk + "e08/200"
        self.Trainer["lr"] = 1e-6
        """

        self.Trainer["ckpt_load_path"] = self.ckpt_disk + "e10/340"
        self.Trainer["lr"] = 1e-6


class e09_8(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        self.RawDataset["scale"] = [4, 8, 8]
        self.ResAttUNet["feats"] = [1, 16, 32, 64, 128, 256, 512]
        self.Evaluer["ckpt_load_path"] = self.ckpt_disk + "e09/340"


class e09(Config):
    def __init__(self) -> None:
        super().__init__("train")
        self.RawDataset["scale"] = [4, 8, 8]
        self.ResAttUNet["feats"] = [1, 16, 32, 64, 128, 256, 512]
        self.Trainer["max_epoch"] = 340
        self.Trainer["ckpt_load_path"] = self.ckpt_disk + "e08/200"
        self.Trainer["lr"] = 1e-6


""" combine reduce scale up list and increase features number
Now we combining the strategy of e03-05 and e06-e07, i.e., we reduce the scale
up list we train from [2, 4, 8, 16] to [4, 8] and increase the features number
of the number from [1, 16, 32] to [1, 16, 32, 64, 128]. 

In e08, we load the ckpt from e03, which is trained with scale up factor 4 and 
luminance off, to train it with scale up factor [4, 8] and lum on.

Result:
Increase complexcity of the network does solve the checkbox problem. In our
prediction of hela cell with scale up 4 in e08, the checkbox completely solved!

features number : [1, 16, 32, 64, 128]
Trainable paras : 1,326,001
Training   speed:      steps /s ( 10 iterations/step)
Validation speed:      steps /s ( 10 iterations/step)
Evaluation speed: 3.02 frames/s ( 16 subframes/frame)
                  4.55 s/frame ( 225 subframes/frame)
"""


class e08_8(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        self.RawDataset["scale"] = [4, 8, 8]
        self.Evaluer["ckpt_load_path"] = self.ckpt_disk + "e08/340"


class e08_4(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        self.Evaluer["ckpt_load_path"] = self.ckpt_disk + "e08/340"


class e08(Config):
    def __init__(self) -> None:
        super().__init__("train")
        """
        self.Trainer["max_epoch"] = 100
        self.Trainer["ckpt_load_path"] = self.ckpt_disk + "e03/10"
        self.Trainer["ckpt_load_lr"] = True
        """

        """
        self.Trainer["max_epoch"] = 200
        self.Trainer["ckpt_load_path"] = self.ckpt_disk + "e08/100"
        self.Trainer["lr"] = 5e-6
        """

        self.Trainer["max_epoch"] = 340
        self.Trainer["ckpt_load_path"] = self.ckpt_disk + "e08/200"
        self.Trainer["lr"] = 1e-6


# All the following configs are no longer maintained


""" reduce scale up list
Now we have two gauss reason:
1.  The network training does not finish, which cause these checkbox.
2.  scale uo to (2, 4, 8, 16) is too expand for the network to learn. 
We will explore the second reason first. We load the result from d04 and
continue to train it with both scale up factor 4 and 8. Then, expand it to
LumT.

Result:
In e11, when scale by 4, the lens of the checkbox is 5; when scale up by 8, the
lens of the checkbox is 9. Thus, the checkbox is in fact cause by super
resolution itself. 

features number : [1, 16, 32]
Trainable paras : 70,353
Training   speed:      steps /s ( 10 iterations/step)
Validation speed:      steps /s ( 10 iterations/step)
Evaluation speed: 3.52 frames/s ( 16 subframes/frame)
                  1.10 frames/s ( 64 subframes/frame)
"""


class e07_8(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        ## data
        self.scale = [4, 8, 8]
        self.ResAttUNet["feats"] = [1, 16, 32]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e07/320"


class e07_4(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        self.ResAttUNet["feats"] = [1, 16, 32]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e07/320"


class e07(Config):
    def __init__(self) -> None:
        super().__init__("train")
        self.ResAttUNet["feats"] = [1, 16, 32]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e06/240"
        self.Trainer["lr"] = 1e-4


class e06(Config):
    def __init__(self) -> None:
        super().__init__("train")
        ## data
        self.lum_info = False
        self.ResAttUNet["feats"] = [1, 16, 32]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "d04/140"
        self.Trainer["lr"] = 1e-4


""" increase feature number
Currently we gauss the problem in e01-02 is because the complexcity of the 
network is not enough to locolize. Reference to d05 and other, it's seems like
these checkbox is cause by not enough training. Thus, we try to increase the
complexcity of the network and retrain.

From 03-05, we define a training process where we start from scale up by 4 LumF,
then expand the salce to (2, 4, 8, 16), and finally LumT. We split first since
it's easier to control the training process of scale up 4 and second since we
may change how we add brightness information in the future.

Result:
In e04_4, (4, 4, 4) LumF, the result is compareable to d04 which is also 
(4, 4, 4) LumF. In e05_4, the problem of checkbox is still exist, like e02_4. 
Thus, we may conclude the complexcity of the network is not the problem.

features number : [1, 16, 32, 64, 128]
Trainable paras : 
Training   speed: 1.07 steps /s ( 10 iterations/step)
Validation speed: 1.70 steps /s ( 10 iterations/step)
Evaluation speed: 2.95 frames/s ( 16 subframes/frame)
"""


class e05_4(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        ## data
        self.lum_info = False
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e05/90"


class e05(Config):
    def __init__(self) -> None:
        super().__init__("train")
        ## data
        self.scale_list = [2, 4, 8, 16]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e04/80"
        self.ckpt_load_lr   = True


class e04_4(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        ## data
        self.lum_info = False
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e04/80"


class e04(Config):
    def __init__(self) -> None:
        super().__init__("train")
        ## SimDataset & RawDataset
        self.lum_info = False
        self.scale_list = [2, 4, 8, 16]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e03/10"
        self.ckpt_load_lr   = True


class e03(Config):
    def __init__(self) -> None:
        super().__init__("train")
        ## SimDataset & RawDataset
        self.lum_info   = False
        self.scale_list = [4]
        ## runner
        self.Trainer["lr"] = 5e-5


""" new architecture
To solve the super resolution problem, we complete redesign the training
process. More specifically, now we fixed the dim_dst to [160 160 160] instead
of variante according to the scale up factor. By fix the input of the network,
the size of the network will independent to the scale up factor. The scale up
factor will only affect how we cut the raw data into subframes to feed into the
network. Also, now we can train various scale up factor at the same time.

In e01, the learning rate is too large, so we reduce it to 1e-5 in e02. 

Result:
In e02_4, we evaluate the network with hela dataset with scale up factor 
[4, 4, 4]. The result show very obvious artifacts with a len 5. The chessbox is 
different from before. 

features number : [1, 16, 32]
Trainable paras : 70,353
Training   speed: 1.14 steps /s ( 10 iterations/step)
Validation speed:      steps /s ( 10 iterations/step)
Evaluation speed: 3.63 frames/s ( 16 subframes/frame)
"""


class e02_4(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        ## data
        self.scale = [4, 4, 4]
        ## model
        self.ResAttUNet["feats"] = [1, 16, 32]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e02/210" 


class e02(Config):
    def __init__(self) -> None:
        super().__init__("train")
        ## data
        self.scale_list = [2, 4, 8, 16]
        ## model
        self.ResAttUNet["feats"] = [1, 16, 32]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "e01/150"


class e01(Config):
    def __init__(self) -> None:
        super().__init__("train")
        ## data
        self.scale_list = [2, 4, 8, 16]
        ## model
        self.ResAttUNet["feats"] = [1, 16, 32]
        ## runner
        self.ckpt_load_path = self.ckpt_disk + "d04/140"
        self.Trainer["lr"] = 1e-4
