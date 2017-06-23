# -*- coding: utf-8 -*-
#
# tools.py
#
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import math
from matplotlib.ticker import MaxNLocator


def combine_images(generated_images):
    """
    Helper function to combine a slew of generated images
    """
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]

    image = np.zeros((height*shape[0], width*shape[1],3),
                     dtype=generated_images.dtype)

    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img

    return image


def plot_normalized_histogram(axis, data, color='r', label='data'):
    """Plot normalized histogram of data
    """
    # Histogram of generated data
    density, bins = np.histogram(data, bins=100,normed=True, density=True)
    unity_density = density / density.sum()
    widths = bins[:-1] - bins[1:]
    axis.bar(bins[1:], unity_density, width=widths, color = color, alpha=0.6, label=label)


def load_csv_data(train_path, test_path):
    """Load and return training and test data from CSV files.

    Requires that there are only a single target value
    """
    # Load training data
    train = np.genfromtxt('./{}'.format(train_path), delimiter=',')
    X_train = train[:, :train.shape[1]-1]
    y_train = train[:, train.shape[1]-1:]

    # Load test data
    test = np.genfromtxt('./{}'.format(test_path), delimiter=',')
    X_test = test[:, :test.shape[1]-1]
    y_test = test[:, test.shape[1]-1:]

    return X_train, y_train, X_test, y_test


def plot_2d_data(X, y, c1='red', c2='blue'):
    """Plots a training set with binary targets.
    """
    plt.scatter(X[np.where(y==0)[0], 0], X[np.where(y==0)[0], 1], color=c1)
    plt.scatter(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], color=c2)


def plot_2d_train_test(X_train, y_train, X_test, y_test):
    """Helper function for plotting train and test set with binary targets.
    """
    plot_2d_data(X_train, y_train, c1='mistyrose', c2='lavender')
    plot_2d_data(X_test, y_test, c1='red', c2='blue')


def load_csv_with_dates(filename, date_idx):
    """Return and set a specific `date_idx` as the DataFrame index.
    """
    data = pd.read_csv(filename, parse_dates=[date_idx])

    print('\nLoading data from `{}`'.format(filename))
    print('Composed of {} lines and {} columns'.format(data.shape[0],
                                                       data.shape[1]))

    return data.set_index([date_idx])


def assess_multivariate_model(model, X_train, y_train, X_test, y_test,
                              test_dates, nb_epochs, batch_size,
                              validation_split):
    """Helper Function to assess multivariate function.

    This function will train the model and do live plots in Jupyter
    notebooks.
    The live plot shows the Losses and the estimation for a test set.
    """
    # Create PLot
    fig,(ax1, ax2) = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.4)

    loss = [] # Keep track of loss
    val_loss = [] # Keep track of validation loss

    # For loop for every 3 epochs
    for epc in tqdm.trange(nb_epochs // 3):

        # Train for 3 epochs and keep track of the history
        EPOCH = model.fit(X_train, y_train,
                          validation_split=validation_split,
                          batch_size=batch_size,
                          epochs=3,
                          verbose=0)

        # Keep track of training and validation error
        loss.extend(EPOCH.history['loss'])
        val_loss.extend(EPOCH.history['val_loss'])

        #predict new points
        pred = model.predict(X_test)

        # Setup ploting properties
        ax1.clear()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.clear()

        # Plot Losses
        ax1.plot(loss,'r', label='loss')
        ax1.plot(val_loss,'g', label='val_loss')
        ax1.legend()
        ax1.grid()

        # Let estimation
        ax2.plot(y_test, '-b', label='data', alpha=0.5)
        ax2.plot(pred, '-r' , label='Prediction', alpha=0.5)
        ax2.legend()

        n_ticks = 20
        ax2.xaxis.set_major_locator(MaxNLocator(n_ticks))
        dates = test_dates
        dates = dates[0:len(y_test):len(y_test)//n_ticks]
        ax2.set_xticklabels(dates, rotation=45)

        # Draw
        fig.canvas.draw()
        time.sleep(0.05)

    return model
