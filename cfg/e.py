from .config import *

__all__ = ["e08", "e09", "e10", "e11",]


""" scale up by 8
Follow the same strategy in e08, we load the ckpt from e08 and continue to train
it with scale up factor 8 and more features. In e09, e10, e11, we try three
different scale up list [4, 8], [8, 16], [4, 8, 16] to see how the scale up list
affect the result. 

Result:
In e09, e11 the checkbox is still exist. In e10, the checkbox is weeker. 

features number : [1, 16, 32, 64, 128, 256, 512]
Trainable paras : 21,401,393
Training   speed: 1.10 steps /s ( 10 iterations/step)
Validation speed: 1.74 steps /s ( 10 iterations/step)
Evaluation speed: 2.90 frames/s ( 16 subframes/frame)
"""


class e11(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["scale_list"] = [4, 8, 16]
        self.validset["scale"] = [4, 8, 8]
        self.model["feats"] = [1, 16, 32, 64, 128, 256, 512]
        self.runner["max_epoch"] = 340
        self.runner["ckpt_load_path"] = "ckpt/e08/200"
        self.runner["lr"] = 1e-6


class e10(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["scale_list"] = [8, 16]
        self.validset["scale"] = [4, 8, 8]
        self.model["feats"] = [1, 16, 32, 64, 128, 256, 512]

        """
        self.runner["max_epoch"] = 340
        self.runner["ckpt_load_path"] = "ckpt/e08/200"
        self.runner["lr"] = 1e-6
        """

        self.runner["max_epoch"] = 450
        self.runner["ckpt_load_path"] = "ckpt/e10/340"
        self.runner["lr"] = 1e-6


class e09(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.validset["scale"] = [4, 8, 8]
        self.model["feats"] = [1, 16, 32, 64, 128, 256, 512]
        self.runner["max_epoch"] = 340
        self.runner["ckpt_load_path"] = "ckpt/e08/200"
        self.runner["lr"] = 1e-6


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


class e08(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        """
        self.runner["max_epoch"] = 100
        self.runner["ckpt_load_path"] = "ckpt/e03/10"
        self.runner["ckpt_load_lr"] = True
        """

        """
        self.runner["max_epoch"] = 200
        self.runner["ckpt_load_path"] = "ckpt/e08/100"
        self.runner["lr"] = 5e-6
        """

        self.runner["max_epoch"] = 340
        self.runner["ckpt_load_path"] = "ckpt/e08/200"
        self.runner["lr"] = 1e-6


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


class e07(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.model["feats"] = [1, 16, 32]
        self.runner["ckpt_load_path"] = "ckpt/e06/240"
        self.runner["lr"] = 1e-4


class e06(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["lum_info"] = False
        self.validset["lum_info"] = False
        self.model["feats"] = [1, 16, 32]
        self.runner["ckpt_load_path"] = "ckpt/d04/140"
        self.runner["lr"] = 1e-4


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


class e05(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["scale_list"] = [2, 4, 8, 16]
        self.runner["ckpt_load_path"] = "ckpt/e04/80"
        self.runner["ckpt_load_lr"]   = True


class e04(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["lum_info"] = False
        self.trainset["scale_list"] = [2, 4, 8, 16]
        self.validset["lum_info"] = False
        self.runner["ckpt_load_path"] = "ckpt/e03/10"
        self.runner["ckpt_load_lr"]   = True


class e03(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["lum_info"] = False
        self.trainset["scale_list"] = [4]
        self.validset["lum_info"] = False
        self.runner["lr"] = 5e-5


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


class e02(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["scale_list"] = [2, 4, 8, 16]
        self.model["feats"] = [1, 16, 32]
        self.runner["ckpt_load_path"] = "ckpt/e01/150"


class e01(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.trainset["scale_list"] = [2, 4, 8, 16]
        self.model["feats"] = [1, 16, 32]
        self.runner["ckpt_load_path"] = "ckpt/d04/140"
        self.runner["lr"] = 1e-4
