# -*- coding: utf-8 -*-
#
# problem_unittests.py
#
import numpy as np


def _print_success_message():
    print('Tests Passed.')
    print('You can move on to the next task.')


def test_build_x(function):
    test_numbers = np.arange(10)
    OUT = function(test_numbers)
    test_shape = (10,2)

    assert type(OUT).__module__ == np.__name__, 'Not NumPy Object'

    assert OUT.shape == test_shape, 'Incorrect Shape. {} shape '\
                                     'found'.format(OUT.shape)

    assert OUT[:, 0].max() == 1 and  OUT[:, 0].min() == 1, 'The first '\
            'columns is not modelling the bias / intercept'

    _print_success_message()


def test_build_y(function):
    test_numbers = np.arange(10)
    OUT = function(test_numbers)
    test_shape = (10,1)


    assert type(OUT).__module__ == np.__name__, 'Not NumPy Object'

    assert OUT.shape == test_shape, 'Incorrect Shape. {} shape '\
                                    'found'.format(OUT.shape)

    _print_success_message()


def test_compute_theta(function):
    XX = np.array([[ 0.,  1.],
                   [ 1.,  1.],
                   [ 2.,  1.],
                   [ 3.,  1.]])
    YY = np.array([[ 1.],
                   [ 3.],
                   [ 5.],
                   [ 7.]])

    OUT = function(XX, YY)
    test_shape = (2, 1)
    A = OUT[0, 0]
    B = OUT[1, 0]

    assert type(OUT).__module__ == np.__name__, 'Not Numpy Object'

    assert OUT.shape == test_shape, 'Incorrect Shape. {} shape '\
                                    'found'.format(OUT.shape)

    assert A - 2.0 < 0.00001 , 'Your function is calculating the wrong '\
                               'slope parameter {}'.format(A)

    assert B - 1.0 < 0.00001 , 'Your function is calculating the wrong '\
                               'intercept parameter {}'.format(B)

    _print_success_message()


def test_simple_model(function):
    nb_inputs = 10
    nb_outputs = 3

    net = function(nb_inputs, nb_outputs)
    weights = net.get_weights()

    assert len(weights) == 2, 'Incorrect Number of Layers.'

    assert weights[0].shape == (nb_inputs, nb_outputs), 'Model has '\
        'the wrong number of inputs and outputs'

    _print_success_message()


def test_simple_model_regularized(function):
    nb_inputs = 10
    nb_outputs = 3

    factor = 20
    net = function(nb_inputs, nb_outputs, factor)
    weights = net.get_weights()

    l = net.get_layer(index=1)
    r = float(l.kernel_regularizer.l2)

    assert abs(r - factor) < 0.00001 , 'Your L2 regularization factor '\
                                       'is not correct : {} instead of '\
                                       '{}'.format(r, factor)

    assert len(weights) == 2, 'Incorrect Number of Layers.'

    assert weights[0].shape == (nb_inputs, nb_outputs), 'Model has '\
        'the wrong number of inputs and outputs'

    _print_success_message()
