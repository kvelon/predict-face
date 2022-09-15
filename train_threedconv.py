import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *

####################
##### Configs ######
####################
data_path = "data/samples_1_sep/augmented.npy"

batch_size = 4
learning_rate = 1e-3
epochs = 100

num_ctx_frames = 1
num_tgt_frames = 1
split_ratio=[0.8, 0.2, 0.0]

model = ThreeDConv_1to9(learning_rate)
liveness_datamodule = LivenessDataModule(batch_size, 
                                         num_ctx_frames, num_tgt_frames,
                                         data_path,
                                         split_ratio=split_ratio)

logger = TensorBoardLogger('logs', 'ThreeDConv_1to9')

trainer = pl.Trainer(gpus=4, 
                     strategy=DDPStrategy(find_unused_parameters=False),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

# trainer = pl.Trainer(gpus=1, 
#                      max_epochs= epochs,
#                      callbacks=LearningRateMonitor(),
#                      logger=logger)

trainer.fit(model, liveness_datamodule)
