# Video Prediction Project for Liveness Data

This repository contains the code required to 1) train three different deep learning architectures to predict liveness data, and 2) plot the frames generated by the trained models.

![Example of a 10-frame video sequence](https://github.com/iProov/predict-weizmann-video/plots/sample_plot.png)


## Downloading data
Liveness data should be downloaded to *./data/* using the functions found in [sci-utils](https://github.com/iProov/sci-utils)

Once the liveness videos are downloaded, run the notebook *preprocess/preprocess.ipynb* to save all the videos in a single numpy array.

This step is optional. You can augment the data by using the notebook *data/augment_data.ipynb*. This notebook will create and save a new numpy array.

## Training
The python scripts prefixed with *train_* are for training the various models. The architectures are implemented in Pytorch and can be found in *./models/*. The hyperparameters for training can be defined in those scripts. For example, in your favourite environment (i.e. conda, Docker), you can run 
```python train_predrnn.py```
to train a PredRNN model. Metrics are automatically logged with a Tensorboard logger.

## Plotting
The Tensorboard logger already contains a sample plot of the predicted frames. If you wish to make further plots, you can use the *plot_5to5_plot1.ipynb*
 notebook to do so. You will need to modify the checkpoint path to load the trained paramaters and weights.