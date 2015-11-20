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
"""Tests for clone method. """

import unittest
import numpy as np

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import lstm_rnn_tied_weights

import theano.tensor as T

from lasagne import layers
from lasagne.layers import InputLayer, EmbeddingLayer, LSTMLayer, SliceLayer

class Test(unittest.TestCase):
    
    def setUp(self):
        pass

    def assert_array_equal(self, *args, **kwargs):
        np.testing.assert_array_equal(*args, **kwargs)

    def test_clone(self):
        # Data for unit testing
        X_unit = ['abcdef', 'abcdef', 'qwerty']
        X_unit = [[ord(c) for c in w] for w in X_unit]
        X_unit = np.array(X_unit, dtype='int8')
        n_alerts_unit, l_alerts_unit = X_unit.shape
        mask_unit = np.ones(X_unit.shape, dtype='int8')

        # Dimensions
        n_alerts = None
        l_alerts = None
        n_alphabet = 2**7 # All ASCII chars
        num_units = 10

        # Symbolic variables
        input_var, input_var2 = T.imatrices('inputs', 'inputs2')
        mask_var, mask_var2 = T.matrices('masks', 'masks2')
        target_var = T.dvector('targets')

        # build net for testing
        l_in = InputLayer(shape=(n_alerts, l_alerts), input_var=input_var, name='INPUT-LAYER')
        l_emb = EmbeddingLayer(l_in, n_alphabet, n_alphabet, 
                                 W=np.eye(n_alphabet),
                                 name='EMBEDDING-LAYER')
        l_emb.params[l_emb.W].remove('trainable') # Fix weight
        l_mask = InputLayer(shape=(n_alerts, l_alerts), input_var=mask_var, name='MASK-INPUT-LAYER')
        l_lstm = LSTMLayer(l_emb, num_units=num_units, name='LSTM-LAYER', mask_input=l_mask)
        l_slice = SliceLayer(l_lstm, indices=-1, axis=1, name="SLICE-LAYER") # Only last timestep

        net = l_slice

        # clone
        l_in2 = InputLayer(shape=(n_alerts, l_alerts), input_var=input_var2, name='INPUT-LAYER2')
        l_mask2 = InputLayer(shape=(n_alerts, l_alerts), input_var=mask_var2, name='MASK-INPUT-LAYER2')
        net2 = lstm_rnn_tied_weights.clone(net, l_in2, l_mask2)
        
        self.assertNotEqual(repr(net), repr(net2))

        pred_unit = layers.get_output(
            net,
            inputs={l_in: input_var, l_mask: mask_var}
        ).eval({input_var: X_unit, mask_var: mask_unit})
        
        pred_unit2 = layers.get_output(
            net2,
            inputs={l_in2: input_var2, l_mask2: mask_var2}
        ).eval({input_var2: X_unit, mask_var2: mask_unit})

        self.assert_array_equal(pred_unit, pred_unit2)


