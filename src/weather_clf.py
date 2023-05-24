print("Starting weather_clf.py ...")

# local tools and their requirements
import sys
sys.path.append("../") # used to allow import from utils folder
import utils.req_functions as rf
import matplotlib.pyplot as plt

# external imports
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping #, ModelCheckpoint
from tensorflow.keras.preprocessing.image import (load_img,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input, ###
                                                 decode_predictions, ###
                                                 VGG16)
from tensorflow.keras.layers import (Flatten,
                                     Dense,
                                     Dropout,
                                     BatchNormalization) ###

# loading image names and assigning labels
def load_image():
    img_dir = os.path.join("..","data","dataset") # path to dataset
    weather_label = ["dew", # labelling the weather categories
                    "fogsmog",
                    "frost",
                    "glaze",
                    "hail",
                    "lightning",
                    "rain",
                    "rainbow",
                    "rime",
                    "sandstorm",
                    "snow"]

    # creating empty lists to fill with for-loop
    label_list = []
    img_list = []

    # specifying paths and labels
    for label in weather_label:
        dir_path = os.path.join(img_dir,label) # create a path to that dir
        for image in os.listdir(dir_path): # for every image file in those folders
            full_path = os.path.join(dir_path,image) # create path to image,
            img_list.append(full_path) # append path to img_list,
            label_list.append(label) # and add weather type to label_list

    df = pd.DataFrame({"img":img_list, "label":label_list}) # create pandas df with image paths in one column and labels in the other

    # making runs briefer for iterative testing purposes, though the full dataset is not too large
    #df = df.sample(frac=0.1, random_state=43) 
    return df, weather_label

# defines label mapping and converts images to usable format
def preprocessing(df):
    label_map = { # encoding labels as integers for label mapping
        "dew" : 0,
        "fogsmog" : 1,
        "frost" : 2,
        "glaze" : 3,
        "hail" : 4,
        "lightning" : 5,
        "rain" : 6,
        "rainbow" : 7,
        "rime" : 8,
        "sandstorm" : 9,
        "snow" : 10
    }

    # mapping the labels to their corresponding integers
    df["label_map"] = df["label"].map(label_map)

    # creating a new array of zeros to fill in
    X = np.zeros((len(df["img"]), 256, 256, 3))

    # reading the images, resizing and rescaling
    for i, img_path in enumerate(df["img"]):
        img = cv2.imread(str(img_path)) # reading each image from its filepath
        img = img/255. # rescaling image values to fractions of 1
        img = cv2.resize(img, (256,256)) # resizing each image to the same size
        X[i] = img # add the resized, rescaled image values at the ith place in the array X

    # assigning the mapped label column (ground truth) to y
    y = df["label_map"]
    return X, y

# split data and labels into training and testing data
def train_test(X,y):
    (X_train, X_test, y_train, y_test) = train_test_split(X,
                                                          y,
                                                          random_state=37, # assigning random state for shuffling
                                                          test_size=0.2, # setting 20% of the data aside for testing
                                                          shuffle=True)

    # converting labels to one-hot encoding to match VGG16's output
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return X_train, X_test, y_train, y_test

# defining the parameters of pre-trained model to be used
def define_model(X_train, y_train):
    # clear keras session and releases previously used models from memory, if any
    tf.keras.backend.clear_session()

    # loading base model
    model = VGG16(include_top=False, # exclude the pre-trained fully connected feed-forward network
                  pooling='avg', # use average values when pooling inputs 
                  input_shape=(256,256,3))

    # mark remaining loaded layers as not trainable, since we want to use the pre-trained ones
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers to replace removed top
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    class2 = Dense(256, activation='relu')(class1)
    class3 = Dense(128, activation='relu')(class2)
    # dropout can help prevent overfitting (important given limited size of dataset) 
    drop1 = Dropout(0.1)(class3)
    output = Dense(11, activation='softmax')(drop1)

    # define new model
    model = Model(inputs=model.inputs, 
                  outputs=output)
###
    # assign learning rates to the weights
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    # compile model according to specified learning rate, loss estimation method, and chosen optimization metric
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# train model within set parameters
def train_model(model, X_train, y_train):
    # outlining requirements for stopping the model early if stops improving (by validation loss per default)
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    early_stopping = [early_stopping] # making list for callbacks

    # save model history as it trains
    model_hist = model.fit(X_train,
                           y_train,
                           validation_split=0.1, # set aside 10% of training data for validation in each batch ###
                           batch_size=128, # how many inputs the neural network trains on before updating its weights ###
                           epochs=50, # 50 epochs should be enough to reach the point of diminishing returns
                           verbose=1,
                           callbacks=early_stopping)
    return model_hist, model

# plot graph of training history using function from utils folder
def plot_graph(model_hist):
    epochs = len(model_hist.history["loss"]) # this is necessary to determine the number of epochs in case early_stopping activates, where a fixed value would fail
    plot = rf.plot_history(model_hist, epochs) # creates and saves plot to out folder
    return None

# saves model to set outpath
def save_model(model_name):
    outpath = os.path.join("..","out","weather_model.keras") # defining outpath for finished model
    
    tf.keras.models.save_model(
        model_name, outpath, overwrite=False, save_format=None, # set overwrite=True to skip console prompt to overwrite old file
    )
    return None

# making classification report using test data
def make_report(model, X_test, y_test, weather_label):
    predictions = model.predict(X_test, batch_size=128) # creating predictions

    # report calls on the ground truth for test data, the predictions, as well as the target labels
    report = classification_report(y_test.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=weather_label)
    return report

# saving classification report to set outpath
def save_report(report):
    outpath = os.path.join("..","out","classification_report.txt") # defining outpath for report

    # overwrite old report file or create new if none exists
    with open(outpath, "w") as file:
        file.write(report)
    return None

def main():
    print("Loading images ...")
    dataframe, weather_label = load_image() # assign images and labels to dataframe
    print("Preprocessing data ...")
    X, y = preprocessing(dataframe) # define label mapping and converts images to usable format
    X_train, X_test, y_train, y_test = train_test(X, y) # train-test split

    print("Defining model parameters ...")
    model = define_model(X_train, y_train) # set new parameters for pre-trained model
    print("Training model ...")
    model_hist, trained_model = train_model(model, X_train, y_train) # train model, assign hist
    save_model(trained_model) # save trained model
    print("Model saved to 'out' folder.")

    plot = plot_graph(model_hist) # make training hist plot and save to out folder
    print("Making classification report ...")
    class_report = make_report(trained_model, X_test, y_test, weather_label) # make classification report on test data
    save_report(class_report) # save report
    print("Report and plot saved to 'out' folder.")
    return None

if __name__ == "__main__":
    main()