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

    def test_split(self):
        a = list(range(10))
        res = list(lstm_rnn_tied_weights.split(a, [60,20,20]))

        print("res="+str(res))

        lens = map(len, res)
        self.assertSequenceEqual(lens, [6, 2, 2])

        from operator import add
        items = sorted(reduce(add, res))
        self.assertSequenceEqual(items, list(range(10)))
