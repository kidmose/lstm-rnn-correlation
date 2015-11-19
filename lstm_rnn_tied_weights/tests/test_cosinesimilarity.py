#!/usr/bin/env python
# Copyright (C) Egon Kidmose 2015-2017
# 
# This file is part of lstm-rnn-correlation.
# 
# lstm-rnn-correlation is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# lstm-rnn-correlation is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with lstm-rnn-correlation. If not, see
# <http://www.gnu.org/licenses/>.
"""Tests for CosineSimilarityLayer. """

import unittest
import numpy as np

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from lstm_rnn_tied_weights import CosineSimilarityLayer

import theano
import theano.tensor as T
from lasagne import layers
from lasagne.layers import InputLayer

class Test(unittest.TestCase):

    def setUp(self):
        test_in_1 = InputLayer((None, None))
        test_in_2 = InputLayer((None, None))
        self.l = CosineSimilarityLayer(test_in_1, test_in_2)
        in1, in2 = T.dmatrices('in1', 'in2')
        pred_out = layers.get_output(
            self.l,
            inputs={test_in_1: in1, test_in_2: in2}
        )
        self.fn = theano.function([in1, in2], pred_out)

    def assert_array_equal(self, *args, **kwargs):
        np.testing.assert_array_equal(*args, **kwargs)


    def test_binary_2d(self):
        in1 = [[0, 1], [1, 0], [0, -1]]
        in2 = [[0, 1], [0, 1], [0, 1]]
        exp = [ 1.,  0., -1.]

        res = self.fn(in1, in2)
        self.assertEqual(
            len(self.l.output_shape),
            len(res.shape),
            msg="Dimension mismatch"
        )
        self.assert_array_equal(exp, res, err_msg="Invalid result")

    def test_int_and_float_2d(self):
        in1 = [[0, 1.1], [10, 0], [0, -2]]
        in2 = [[0, 0.9], [0, 0.1], [0, 3]]
        exp = [1.,  0., -1.]

        res = self.fn(in1, in2)
        self.assertEqual(
            len(self.l.output_shape),
            len(res.shape),
            msg="Dimension mismatch"
        )
        self.assert_array_equal(exp, res, err_msg="Invalid result")

    def test_int_1d(self):
        in1 = [[1], [-1]]
        in2 = [[1], [1]]
        exp = [1., -1]

        res = self.fn(in1, in2)
        self.assertEqual(
            len(self.l.output_shape),
            len(res.shape),
            msg="Dimension mismatch"
        )
        self.assert_array_equal(exp, res, err_msg="Invalid result")
