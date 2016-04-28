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
"""Tests script. """

import unittest

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import lstm_rnn_tied_weights

from subprocess import call

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        if os.path.isfile('tmp.py'):
            os.remove('tmp.py')

    def test_exported(self):
        bn = 'lstm-rnn-tied-weights'
        r = call([
            'ipython', 'nbconvert',
            '--to', 'script',
            '--output', 'tmp',
            bn+'.ipynb'
        ])
        self.assertFalse(r, "Failed to export")
        r = call(['diff', bn+'.py', 'tmp.py'])
        self.assertFalse(r, 'script and notebook mismatch')

    def script_helper(
            self,
            **kwargs
    ):
        env = os.environ.copy()
        env['MAX_PAIRS'] = '100'
        env['BATCH_SIZE'] = '2'
        env['NN_UNITS'] = '2'
        env['CUT'] = 'none'
        env['CLUSTER_SAMPLES'] = '100'
        env['TEST'] = 'True'
        for k, v in kwargs.items():
            env[k] = v
        self.assertFalse(call([sys.executable, 'lstm-rnn-tied-weights.py'], env=env))

    def test_cut_none(self):
        self.script_helper(CUT='none')

    def test_cut_inc(self):
        self.script_helper(
            CUT='inc',
            SPLIT='1,1,1',
        )

    def test_cut_alert(self):
        self.script_helper(CUT='alert')

    def test_cut_pair(self):
        self.script_helper(CUT='pair')

    def test_mask_ip(self):
        self.script_helper(MASKING='ip')

    def test_mask_ts(self):
        self.script_helper(MASKING='ts')

    def test_mask_tsip(self):
        self.script_helper(MASKING='tsip')

    def test_nn_units(self):
        self.script_helper(NN_UNITS='1')
        self.script_helper(NN_UNITS='1,2,3')

    def test_learning_rate(self):
        self.script_helper(NN_LEARNING_RATE='1e-6')
        self.script_helper(NN_LEARNING_RATE='1e6')
