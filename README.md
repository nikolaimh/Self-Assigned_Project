# Assignment 4 – Self-Assigned Project

## 1.	Contributions
This assignment took inspiration from the in-class notebooks and uses the same req_functions.py with code written by Ross as mentioned above. This Medium article was also used, particularly for understanding the directory structure of my data and for transferring labels to a dataframe, and ChatGPT was applied to understand the errors I encountered.

The weather picture dataset was found here, uploaded to the public domain by Kaggle user Jehan Bhathena who, in turn, cites Haixia Xiao for the Weather Phenomenon Database. 

## 2.	Methods
The image names are loaded as strings and assigned class labels from the directory in which each image was stored, and a dataframe is created containing image name, label, and label map. From this, the image data itself is loaded, resized, and rescaled. The images are split into testing and training segments and the label maps are binarized.

The pretrained base model (VGG16) is then loaded without its top layers, and new top layers are defined. Learning rate is set and the model is compiled and trained using the training data with 10% of the data set aside for validation. The model trains for 50 epochs, with early stopping enabled if the model does not improve for five epochs in a row, at which point the training history is plotted and a classification report is made based on test data predictions. Both of these are saved to the out folder, along with the model itself.

## 3.	Usage
Download the dataset from https://www.kaggle.com/datasets/jehanbhathena/weather-dataset and place it into the data directory. The script will expect the subfolder containing the images to be named dataset.

Before running the script, the requisite packages must be installed. To do this, ensure that the current directory is Self-Assigned_Project and execute bash setup.sh from the command line, which will update pip and install the packages noted in requirements.txt.

In the same directory, execute python src/weather_clf.py from command line to run the script. To run the script on a fraction of the full dataset, comment line 63 back in before running and determine the fraction there. The number of epochs is determined on line 165, currently set to 50.

## 4.	Discussion
