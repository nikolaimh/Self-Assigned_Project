# Assignment 4 – Self-Assigned Project

## 1.	Contributions
This assignment took inspiration from the in-class notebooks and uses the same ```req_functions.py``` with code written by Ross as in Assignment 3. This [Medium article](https://medium.com/mlearning-ai/image-classification-for-beginner-a6de7a69bc78) was also used, particularly for understanding the directory structure of my data and for transferring labels to a dataframe, and ChatGPT was applied to understand the errors I encountered.

The weather picture dataset was found [here](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset), uploaded to the public domain by Kaggle user Jehan Bhathena who, in turn, cites Haixia Xiao for the Weather Phenomenon Database. 

## 2.	Methods
The image names are loaded as strings and assigned class labels from the directory in which each image was stored, and a dataframe is created containing image name, label, and label map. From this, the image data itself is loaded, resized, and rescaled. The images are split into testing and training segments and the label maps are binarized.

The pretrained base model ```VGG16``` is then loaded without its top layers, and new top layers are defined. Learning rate is set and the model is compiled and trained using the training data with 10% of the data set aside for validation. The model trains for 50 epochs, with early stopping enabled if the model does not improve for five epochs in a row, at which point the training history is plotted and a classification report is made based on test data predictions. Both of these are saved to the ```out``` folder, along with the model itself.

## 3.	Usage
Download the dataset from the link in Contributions and place it into the data directory. The script will expect the subfolder containing the images to be named dataset. Two of the ```.jpg``` files, ```4514.jpg``` and ```1187.jpg```, turned out to be gifs and were manually removed before running.

Before running the script, the requisite packages must be installed. To do this, ensure that the current working directory is ```Self-Assigned_Project``` and execute ```bash setup.sh```from the command line, which will update ```pip``` and install the packages noted in ```requirements.txt```.

In the same directory, execute ```python src/weather_clf.py``` from command line to run the script. To run the script on a fraction of the full dataset, comment line 63 back in before running and determine the fraction there. The number of epochs is determined on line 165, currently set to 50.

## 4.	Discussion
The fully trained model, currently saved to the ```out``` folder, achieved a fine result considering the training time (<4 hours on a 32 core vCPU on UCloud), with training and validation accuracy reaching circa 70%. This is underscored by the test predictions yielding an average f1 accuracy of 74%, with several categories scoring in the high 80s. Neither the loss nor the accuracy curves appear to have entirely reached the point of diminishing returns, indicating that more epochs would have yielded significantly better results.

These results are probably in large part due to the relatively small size of the dataset, with the number of images reaching 6880 rather than the tens or hundreds of thousands often seen for using pretrained CNNs. Image generation and alteration of present data could have alleviated this, but time with large UCloud vCPUs has been limited in recent weeks. A dropout rate of 0.1 was added to combat overfitting, however, and it appears to have worked considering the favorable test predictions.

Ultimately, this points to weather identification being a relatively simple task for a pretrained CNN, likely due to it depending more on color distributions as it does precise shapes. Given a larger dataset (generated or not) and a longer training time, had vCPU time been available, the model may have reached much higher.

![Training History](https://github.com/nikolaimh/Self-Assigned_Project/blob/main/out/weather_plot.png)

|             | precision   | recall | f1-score  | support|
|-------------|-------------|--------|-----------|--------|
|         dew |      0.88   |   0.90 |     0.89  |     136|
|     fogsmog |      0.81   |   0.73 |     0.76  |     183|
|       frost |      0.73   |   0.61 |     0.66  |     104|
|       glaze |      0.59   |   0.66 |     0.62  |     138|
|        hail |      0.90   |   0.83 |     0.86  |     121|
|   lightning |      0.88   |   0.87 |     0.88  |      69|
|        rain |      0.80   |   0.60 |     0.69  |      93|
|     rainbow |      1.00   |   0.55 |     0.71  |      51|
|        rime |      0.70   |   0.90 |     0.78  |     229|
|   sandstorm |      0.58   |   0.69 |     0.63  |     124|
|        snow |      0.64   |   0.56 |     0.60  |     124|
||
|    accuracy |             |        |     0.74  |    1372|
|   macro avg |      0.77   |   0.72 |     0.73  |    1372|
|weighted avg |      0.75   |   0.74 |     0.74  |    137|2
