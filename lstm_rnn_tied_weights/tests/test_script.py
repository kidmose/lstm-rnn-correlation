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
        call(['rm', 'tmp.py'])

    def test_mask(self):
        env = os.environ.copy()
        env['MAX_PAIRS'] = '10'
        env['BATCH_SIZE'] = '2'

        self.assertFalse(call([sys.executable, 'lstm-rnn-tied-weights.py'], env=env))
        env['MASKING'] = 'ip'
        self.assertFalse(call([sys.executable, 'lstm-rnn-tied-weights.py'], env=env))
        env['MASKING'] = 'ts'
        self.assertFalse(call([sys.executable, 'lstm-rnn-tied-weights.py'], env=env))
        env['MASKING'] = 'tsip'
        self.assertFalse(call([sys.executable, 'lstm-rnn-tied-weights.py'], env=env))

    def cut_helper(self, env_cut):
        env = os.environ.copy()
        env['MAX_PAIRS'] = '10'
        env['BATCH_SIZE'] = '2'
        env['CUT'] = env_cut
        self.assertFalse(call([sys.executable, 'lstm-rnn-tied-weights.py'], env=env))

    def test_cut_none(self):
        self.cut_helper('none')

    def test_cut_inc(self):
        self.cut_helper('inc')

    def test_cut_alert(self):
        self.cut_helper('alert')

    def test_cut_pair(self):
        self.cut_helper('pair')

