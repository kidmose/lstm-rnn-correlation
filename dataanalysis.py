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

from __future__ import print_function
from __future__ import division

import logging
logging.getLogger().handlers = []

import sys
import os
import time
import glob
import json
import re
import subprocess
from functools import reduce
from operator import itemgetter

import numpy as np

import lstm_rnn_tied_weights
from lstm_rnn_tied_weights import CosineSimilarityLayer
from lstm_rnn_tied_weights import load, encode, cross_join, iterate_minibatches
logger = lstm_rnn_tied_weights.logger

l = load(glob.glob('data/*.out'))
incidents, alerts = zip(*[(i, item) for i, sublist in l for item in sublist])

unique, counts = np.unique(incidents, return_counts=True)

print('\ncounts:')
def count():
    print(np.asarray((unique, counts)).T)

count()

incs = dict()
for u in unique:
    incs[u] = [a for i, a in zip (incidents, alerts) if i == u]


def disp(incident, max_lines=10):
    res = incs[incident]
    if len(res) > max_lines:
        res = res[:max_lines-1] + ['...'] + [res[-1]]
    for a in res:
        print(a.strip())

for i in range(1,12):
    print('### {} ###'.format(i))
    print()
    disp(i)
    print()
    print()

def get_ips(s):
    return re.findall('(?:[0-9]{1,3}\.){3}[0-9]{1,3}', s)


print(" ### IPs seen in incidents ### ")
for incident, alerts in incs.items():
    print("incident: {}".format(incident))
    ips = (reduce(list.__add__, map(get_ips, alerts)))
    u, c = np.unique(ips, return_counts=True)
    cnts = zip(list(u), list(c))
    cnts = sorted(cnts, key=itemgetter(1), reverse=True)
    print(list(cnts))

from datetime import datetime
print(" ### Timeline ### ")
events = list()
for incident, alerts in incs.items():
    def time_str(alert):
        return str(datetime.strptime(alert.split()[0], '%m/%d-%H:%M:%S.%f'))
    events.append((time_str(alerts[0]), 'start', incident))
    events.append((time_str(alerts[-1]), 'end', incident))
for e in sorted(events):
    print(e)

    
    


    
