
# coding: utf-8

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

# # Alert correlation with a Long Short-Term Memoroy (LSTM) Recurrent Neural Network(RNN) and cosine similarity
# 
# **Author:** Egon Kidmose, egk@es.aau.dk
# 
# In network security a common task is to detect network intrusions and for this purpose an Intrusion Detections System (IDS) can be used to raise alerts on suspicious network traffic.
# Snort, Suricata and Bro are examples of free and open source IDSs (Commercial products also exist).
# The alerts generally provides low level information such as recognition of strings that are known to be part of security exploits or anomalous connection rates for a host.
# By grouping alerts that are correlated into higher level events, false positives might be suppressed and attack scenarios becomes easier to recognise.
# This is a take on how to correlate IDS alerts to determine which belong in the same group.
# 
# Alerts can be represented as log lines with various information such as time stamp, IP adresses, protocol information and a description of what triggered the alert.
# It is assumed that such a log lines hold the information needed to determine if two alerts are correlated or not.
# 
# The input to the neural network will be two alerts and the output will indicate if they are correlated or not.
# In further detail the inputs is two strings of ASCII characters of variable length.
# For the output a Cosine Similarity layer is implemented and used to produce an output in the range [-1,1], with -1 meaning opposite, 0 meaning orthogonal and 1 meaning the same.
# 
# For the hidden layers only a single layers of Long Short-Term Memory (LSTM) cells is used.
# It is an option to experiment with adding more.
# Being reccurrent, such a layer handles variable length input well.
# 
# While it turned out to be to challenging to implement, the initial idea was to let the two inputs pass through LSTM layers with identical weights.
# The intent was to have them act as transformations into a space where cosine similarity could be used to measure similarity of the alerts.
# However I have not succeded at tying the weights together.
# As an alternative this might be achieved by using all training pairs in both original and swapped order.
# The intuition is that this leads to two identical layers, but intuition also suggest that this is highly ineffective.
# 
#                       Output
#                         |
#     Cosine similarity   #
#                        / \
#         LSTM layers   #   #
#                       |   |
#         "alert number 1"  |
#             "alert number 2"
# 
# 
# Reference: Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data." Proceedings of the 22nd ACM international conference on Conference on information & knowledge management. ACM, 2013.
# 

# In[ ]:

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

import netaddr
import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.objectives import *

import lstm_rnn_tied_weights
from lstm_rnn_tied_weights import CosineSimilarityLayer
from lstm_rnn_tied_weights import load_data, encode, split_data, cross_join, iterate_minibatches
logger = lstm_rnn_tied_weights.logger

env = dict()
# git
env['version'] = subprocess.check_output(["git", "describe"]).strip()
if not isinstance(env['version'], str):
    env['version'] = str(env['version'], "UTF-8")

# OMP
env['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', str())

# Masking
env['MASKING'] = os.environ.get('MASKING', str())
env['MASK_IP'] = 'ip' in env['MASKING'].lower()
env['MASK_TS'] = 'ts' in env['MASKING'].lower()

# cutting
env['CUTTING'] = os.environ.get('CUTTING', str())
if 'pair' in env['CUTTING'].lower():
    env['CUT_PAIR'] = True
elif 'incident' in env['CUTTING'].lower():
    env['CUT_INC'] = True
else:
    env['CUT_NO'] = True

# Data control
env['MAX_PAIRS'] = int(os.environ.get('MAX_PAIRS', 1000000))
env['BATCH_SIZE'] = int(os.environ.get('BATCH_SIZE', 10000))
env['EPOCHS'] = int(os.environ.get('EPOCHS', 10))
env['SPLIT'] = [int(el) for el in os.environ.get('SPLIT', '60,20,20').split(',')]

logger.info("Starting.")
logger.info("env: " + str(env))
for k in sorted(env.keys()):
    logger.info('env[\'{}\']: {}'.format(k,env[k]))


# In[ ]:

# Data for unit testing
X_unit = ['abcdef', 'abcdef', 'qwerty']
X_unit = [[ord(c) for c in w] for w in X_unit]
X_unit = np.array(X_unit, dtype='int8')
logger.debug(X_unit)
n_alerts_unit, l_alerts_unit = X_unit.shape
mask_unit = np.ones(X_unit.shape, dtype='int8')
logger.debug(mask_unit)


# In[ ]:

# Dimensions
n_alerts = None
l_alerts = None
n_alphabet = 2**7 # All ASCII chars
num_units = 10


# In[ ]:

# Symbolic variables
input_var, input_var2 = T.imatrices('inputs', 'inputs2')
mask_var, mask_var2 = T.matrices('masks', 'masks2')
target_var = T.dvector('targets')


# In[ ]:

# First line
l_in = InputLayer(shape=(n_alerts, l_alerts), input_var=input_var, name='INPUT-LAYER')
l_emb = EmbeddingLayer(l_in, n_alphabet, n_alphabet,
                         W=np.eye(n_alphabet),
                         name='EMBEDDING-LAYER')
l_emb.params[l_emb.W].remove('trainable') # Fix weight
l_mask = InputLayer(shape=(n_alerts, l_alerts), input_var=mask_var, name='MASK-INPUT-LAYER')
l_lstm = LSTMLayer(l_emb, num_units=num_units, name='LSTM-LAYER', mask_input=l_mask)
l_slice = SliceLayer(l_lstm, indices=-1, axis=1, name="SLICE-LAYER") # Only last timestep
net = l_slice


# In[ ]:

# Test first line

# Test InputLayer
pred_unit = get_output(l_in, inputs={l_in: input_var}).eval(
    {input_var: X_unit})
assert (pred_unit == X_unit).all(), "Unexpected output"
# Test EmbeddingLayer
pred_unit = get_output(l_emb, inputs={l_in: input_var}).eval(
    {input_var: X_unit})
assert (np.argmax(pred_unit, axis=2) == X_unit).all()
assert np.all(pred_unit.shape == (n_alerts_unit, l_alerts_unit, n_alphabet ))
# Test LSTMLayer
pred_unit = get_output(
    l_lstm,
    inputs={l_in: input_var, l_mask: mask_var}
).eval({input_var: X_unit, mask_var: mask_unit})
assert pred_unit.shape == (n_alerts_unit, l_alerts_unit, num_units), "Unexpected dimensions"
pred_unit = get_output(
    l_lstm,
    inputs={l_in: input_var, l_mask: mask_var}
).eval({input_var: [[1],[1]], mask_var: [[1],[1]]})
assert np.all(pred_unit[0] == pred_unit[1]), "Repeated alerts must produce the same"
pred_unit = get_output(
    l_lstm,
    inputs={l_in: input_var, l_mask: mask_var}
).eval({input_var: [[1,1],[1,1]], mask_var: [[1,1],[1,1]]})
assert np.all(pred_unit[0] == pred_unit[1]), "Repeated alerts must produce the same"
pred_unit = get_output(
    l_lstm,
    inputs={l_in: input_var, l_mask: mask_var}
).eval({input_var: [[1,1],[0,1]], mask_var: [[1,1],[1,1]]})
assert np.all(pred_unit[0] != pred_unit[1]), "Earlier must affect laters"
pred_unit = get_output(
    l_lstm,
    inputs={l_in: input_var, l_mask: mask_var}
).eval({input_var: [[1,0],[1,1]], mask_var: [[1,1],[1,1]]})
assert np.all(pred_unit[0,0] == pred_unit[1,0]), "Later must not affect earlier"
assert np.all(pred_unit[0,1] != pred_unit[1,1]), "Current must make a difference"
# Test SliceLayer
pred_unit = get_output(
    l_slice,
    inputs={l_in: input_var, l_mask: mask_var}
).eval({input_var: X_unit, mask_var: mask_unit})
assert pred_unit.shape == (n_alerts_unit, num_units), "Unexpected shape"
pred_unit_lstm = get_output(
    l_lstm,
    inputs={l_in: input_var, l_mask: mask_var}
).eval({input_var: X_unit, mask_var: mask_unit})
assert np.all(pred_unit_lstm[:, -1, :] == pred_unit), "Unexpected result of slicing"

logger.debug('OK')


# In[ ]:

# Second line as a copy with shared weights
l_in2 = InputLayer(shape=l_in.shape, input_var=input_var2, name=l_in.name+'2')
l_mask2 = InputLayer(shape=l_mask.shape, input_var=mask_var2, name=l_mask.name+'2')
net2 = lstm_rnn_tied_weights.clone(net, l_in2, l_mask2)


# In[ ]:

# Merge lines
l_cos = CosineSimilarityLayer(net, net2, name="COSINE-SIMILARITY-LAYER")
l_sig = NonlinearityLayer(l_cos, nonlinearity=sigmoid, name="SIGMOID-LAYER")
cos_net = l_sig


# In[ ]:

# Training Procedure
t = time.time()
prediction = get_output(cos_net)
loss = binary_crossentropy(prediction, target_var)
loss = loss.mean()
params = get_all_params(cos_net, trainable=True)
updates = sgd(loss, params, learning_rate=0.1)

# Testing Procedure
test_prediction = get_output(cos_net, deterministic=True)
test_loss = binary_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(test_prediction > 0.5, target_var),
                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, input_var2, mask_var, mask_var2, target_var], loss, updates=updates)
val_fn = theano.function([input_var, input_var2, mask_var, mask_var2, target_var], [test_loss, test_acc])
logger.debug("Spent {}s compilling.".format(time.time()-t))


# In[ ]:

def get_random_ip(invalid={}, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ip = netaddr.IPAddress('.'.join(
            [str(octet) for octet in np.random.randint(256, size=(4))]
        ))

    if str(ip) not in invalid and ip.is_unicast() and not ip.is_loopback():
        return str(ip)
    else:
        return get_random_ip(invalid=invalid)

def mask_ips(alerts):
    logger.info('Masking out IP addresses')
    pattern = '(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    fn = lambda alert : re.sub(pattern, 'IP', alert)
    return map(fn, alerts)

def mask_tss(alerts):
    logger.info('Masking out timestamps')
    pattern = '[0-9]{2}/[0-9]{2}-[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}'
    fn = lambda alert : re.sub(pattern, 'TIMESTAMP', alert)
    return map(fn, alerts)

def break_down_data(labels):
    u, c = np.unique(labels, return_counts=True)
    c_norm = (c/sum(c))*100
    result = str()
    result += 'Label: '+('{: >10}'*len(u)).format(*u) + '\n'
    result += 'Count: '+('{: >10}'*len(u)).format(*c) + '\n'
    result += 'Norm.: '+('{: >9.2f}%'*len(u)).format(*c_norm)
    return result

data = load_data(glob.glob('data/*.out'))

logger.info("Rewriting victim IP address")
old_ip = '147.32.84.165'
used_ips = {old_ip}
for incident, alerts in data.items():
    new_ip = get_random_ip(invalid=used_ips, seed=incident)
    used_ips.add(new_ip)
    logger.info("Replacing {} with {} for incident {}".format(
            old_ip, new_ip, incident,
        ))
    data[incident] = map(lambda alert: alert.replace(old_ip, new_ip), alerts)

incidents = list()
alerts = list()
for k,vs in data.items():
    for v in vs:
        incidents.append(k)
        alerts.append(v)

# If/what to mask out
if env['MASK_IP']:
    alerts = mask_ips(alerts)
if env['MASK_TS']:
    alerts = mask_tss(alerts)

data = encode(alerts, incidents)
logger.info('Breakdown of original data:\n'+break_down_data(incidents))

# How to split train/validation/test data
if env.get('CUT_PAIR', False):
    def _get_batch(offset):
        pairs = cross_join(data, offset=offset)
        for i in range(env['MAX_PAIRS']):
            yield next(pairs)

    get_train_batch = lambda : _get_batch(0)
    get_val_batch = lambda : _get_batch(env['MAX_PAIRS'])
    get_test_batch = lambda : _get_batch(env['MAX_PAIRS']*2)

elif env.get('CUT_INC', False):
    alerts, masks, incidents = data
    train, val, test = split_data(alerts, masks, incidents, env['SPLIT'])

    def _get_batch(batch):
        for sample in cross_join(batch, max_alerts=env['MAX_PAIRS']):
            yield sample

    get_train_batch = lambda : _get_batch(train)
    get_val_batch = lambda : _get_batch(val)
    get_test_batch = lambda : _get_batch(test)

elif env.get('CUT_NO', False):
    def get_train_batch():
        for sample in cross_join(data, max_alerts=env['MAX_PAIRS']):
            yield sample

    get_val_batch = get_train_batch
    get_test_batch = get_train_batch

list(get_train_batch())[0]

a1, a2, m1, m2, cor, inc1, inc2 = range(7)
logger.info('Breakdown of training data, correlation:\n'+
            break_down_data([p[cor] for p in get_train_batch()])
           )
logger.info('Breakdown of training data, incident 1:\n'+
            break_down_data([p[inc1] for p in get_train_batch()])
           )
logger.info('Breakdown of training data, incident 2:\n'+
            break_down_data([p[inc2] for p in get_train_batch()])
           )

logger.info("Starting training...")
for epoch in range(env['EPOCHS']):
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(get_train_batch(), env['BATCH_SIZE']):
        train_err += train_fn(*batch)
        train_batches += 1
        logger.debug('Batch complete')

    #if (epoch+1) % (env['EPOCHS']/10) == 0:
    if True:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(get_val_batch(), env['BATCH_SIZE']):
            err, acc = val_fn(*batch)
            val_err += err
            val_acc += acc
            val_batches += 1

        logger.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, env['EPOCHS'], time.time() - start_time))
        logger.info("  training loss:\t\t{:.20f}".format(train_err / train_batches))
        logger.info("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        logger.info("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    logger.debug('Epoch complete')

test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(get_test_batch(), env['BATCH_SIZE']):
    err, acc = val_fn(*batch)
    test_err += err
    test_acc += acc
    test_batches += 1

logger.info("Final results:")
logger.info("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
logger.info("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))

model = {'model':{str(p): v.tolist() for p, v in zip(get_all_params(cos_net), get_all_param_values(cos_net))}}
model_str = json.dumps(model)
logger.debug('Dumping model parameters:')
logger.debug(model_str)



logger.info('Completed.')


# In[ ]:



