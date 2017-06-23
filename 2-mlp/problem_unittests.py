# -*- coding: utf-8 -*-
#
# problem_unittests.py
#
import numpy as np


def _print_success_message():
    print('Tests Passed.')
    print('You can move on to the next task.')


def test_forward_pass(function, a1, a2, a3):
    vals = function(np.arange(10))
    correct_logistic = np.array([0.5,
                                 0.73105858,
                                 0.88079708,
                                 0.95257413,
                                 0.98201379,
                                 0.99330715,
                                 0.99752738,
                                 0.99908895,
                                 0.99966465,
                                 0.99987661])

    correct_u = np.array([0.97587298, 0.66818777, 0.04310725])
    correct_v = np.array([0.57056101, 0.33543499, 0.56179933, 0.6315474])
    correct_y = np.array([0.66400074])

    assert np.allclose(correct_logistic, vals), 'The logistic function '\
        'is not implemented correctly.'

    assert np.allclose(correct_u, a1), 'The activation of layer `U` has '\
        'not been calculated correctly.'
    assert np.allclose(correct_v, a2), 'The activation of layer `V` has '\
        'not been calculated correctly.'
    assert np.allclose(correct_y, a3), 'The activation of layer `y` has '\
        'not been calculated correctly.'

    _print_success_message()


def test_xor_model(function):
    net = function()
    weights = net.get_weights()

    assert len(weights) == 4, 'Incorrect number of Layers or missing '\
        'bias neurons.'

    assert weights[0].shape == (2, 2), 'Model has the wrong number of '\
        'weights from input to hidden layer.'

    assert weights[1].shape == (2,), 'Model has the wrong number of '\
        'bias weights in the hidden layer.'

    assert weights[2].shape == (2, 1), 'Model has the wrong number of '\
        'weights from hidden layer to output layer.'

    assert weights[3].shape == (1,), 'Model has the wrong number of '\
        'bias weights in the output layer.'

    _print_success_message()
