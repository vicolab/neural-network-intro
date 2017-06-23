====================================================
Practical Introduction to Artificial Neural Networks
====================================================

This repository contains a practical introduction to artificial neural networks by way of `Jupyter notebooks`_ and the `Python programming language`_.

This mini-course was prepared by Igor Barros Barbosa and Aleksander Rognhaugen.


Structure
=========

The mini-course is split up into the following components:

0. `Getting started with NumPy`_ [optional]
1. Regression
    a) Linear regression
    b) Multivariate linear regression
2. Multilayer perceptron
3. Convolutional neural networks
4. Generative adversarial networks

The zeroth component refers to an optional getting started guide for those of you not too familiar with NumPy arrays, e.g. vectors, matrices. We strongly recommend that you scan through it and use it as a reference for whenever you need it in any of the other Jupyter notebooks.

All of the notebooks use the high-level artificial neural network package `Keras`_ for creating and training machine learning models. The notebooks assume you are using `TensorFlow`_.


Running Jupyter
===============

To run the Jupyter notebooks on your local machine you will have to (i) clone the repository and (ii) start Jupyter via the terminal using the following command:

.. code-block:: bash

  jupyter notebook

When Jupyter is running, you can point your web browser at ``http://localhost:8888`` to start using the notebook. The start screen displays all available Jupyter notebooks in the current directory. You may have to change directory to find the notebook you want to read.

Press **Shift-Enter** to run the currently selected cell.


Python Packages
===============

The notebooks assume you have the following Python packages installed

+-----------------+------------------------------------------------------------------------------------+
| Library         | Function                                                                           |
+=================+====================================================================================+
| `Jupyter`_      | Installs all necessary components of the `Jupyter`_ system                         |
+-----------------+------------------------------------------------------------------------------------+
| `NumPy`_        | Adds support for multi-dimensional arrays along with algorithms to operate on them |
+-----------------+------------------------------------------------------------------------------------+
| `Pillow`_       | A PIL (Python Imaging Library) fork                                                |
+-----------------+------------------------------------------------------------------------------------+
| `matplotlib`_   | A 2D plotting package                                                              |
+-----------------+------------------------------------------------------------------------------------+
| `Pandas`_       | A data analysis package                                                            |
+-----------------+------------------------------------------------------------------------------------+
| `tqdm`_         | A package for easily displaying progress meters                                    |
+-----------------+------------------------------------------------------------------------------------+
| `TensorFlow`_   | A numerical computing library using data flow graphs                               |
+-----------------+------------------------------------------------------------------------------------+
| `Keras`_        | A high-level artificial neural network package                                     |
+-----------------+------------------------------------------------------------------------------------+

For those of you with `pip` see `requirements.txt`.


.. Links

.. _Jupyter notebooks: http://jupyter.org/
.. _Python programming language: https://www.python.org/
.. _Getting started with NumPy: https://github.com/vicolab/tdt4195-public/blob/master/digital-image-processing/getting-started/getting-started-python.ipynb
.. _Keras: https://keras.io/
.. _TensorFlow: https://www.tensorflow.org/
.. _Jupyter: http://jupyter.org/
.. _NumPy: http://www.numpy.org/
.. _Pillow: https://python-pillow.org/
.. _matplotlib: http://matplotlib.org/
.. _Pandas: http://pandas.pydata.org/
.. _tqdm: https://pypi.python.org/pypi/tqdm
