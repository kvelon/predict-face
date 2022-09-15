import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *

####################
##### Configs ######
####################
data_path = "data/samples_1_sep_unfiltered/combined.npy"

batch_size = 2
learning_rate = 1e-3
epochs = 100

num_ctx_frames = 1
num_tgt_frames = 9
split_ratio=[0.8, 0.2, 0.0]

input_channels=3
num_hidden=[64, 64, 64]
kernel_size=5
stride=1
learning_rate=1e-3

model = PredRNN(input_channels=input_channels,
                num_hidden=num_hidden,
                num_ctx_frames=num_ctx_frames,
                num_tgt_frames=num_tgt_frames,
                kernel_size=kernel_size,
                stride=stride,
                learning_rate=learning_rate)

liveness_datamodule = LivenessDataModule(batch_size, 
                                         num_ctx_frames, num_tgt_frames,
                                         data_path,
                                         split_ratio=split_ratio)

logger = TensorBoardLogger('./logs', 'PredRNN')

trainer = pl.Trainer(gpus=2, 
                     strategy=DDPStrategy(find_unused_parameters=False),
                     num_sanity_val_steps=0,
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, liveness_datamodule)                                         