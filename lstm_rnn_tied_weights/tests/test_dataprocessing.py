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
"""Tests for data processing functions. """

import unittest
import numpy as np

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import lstm_rnn_tied_weights

from collections import Iterable

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def assert_array_equal(self, *args, **kwargs):
        def it(a, b, lvl=''):
            """Walks two compound Iterable as a trees and ensures equality"""
            print(lvl + 'types: {} {}'.format(type(a), type(b)))
            if type(a) != type(b):
                raise AssertionError
            if isinstance(a, Iterable):
                if len(a) != len(b):
                    raise AssertionError
                for aa, bb in zip(a,b):
                    it(aa, bb, lvl=lvl+' ')
            else:
                if a != b:
                    raise AssertionError
                print(lvl + '{} {}'.format(a,b))

        it(args[0], args[1], lvl='\t')

    def test_split_data(self):
        a = np.zeros((10, 1), dtype=int)
        res = list(lstm_rnn_tied_weights.split_data(a, a+1, a+2, [60,20,20]))
        exp = [
            (np.array([[0]]*6), np.array([[1]]*6), np.array([[2]]*6)),
            (np.array([[0]]*2), np.array([[1]]*2), np.array([[2]]*2)),
            (np.array([[0]]*2), np.array([[1]]*2), np.array([[2]]*2)),
        ]
        self.assert_array_equal(res, exp)
