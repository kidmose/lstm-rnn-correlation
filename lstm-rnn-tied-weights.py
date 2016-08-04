
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
import subprocess
import datetime
import socket

from operator import itemgetter

import numpy as np
import scipy as sp
import theano
import theano.tensor as T
import matplotlib
import pandas as pd

import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.objectives import *

import lstm_rnn_tied_weights
from lstm_rnn_tied_weights import CosineSimilarityLayer
from lstm_rnn_tied_weights import load, modify, split, pool, cross_join, limit, break_down_data
from lstm_rnn_tied_weights import iterate_minibatches, encode
from lstm_rnn_tied_weights import mask_ips, mask_tss, mask_ports
from lstm_rnn_tied_weights import uniquify_victim, extract_prio, get_discard_by_prio
logger = lstm_rnn_tied_weights.logger

runid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + socket.gethostname()
out_dir = 'output/' + runid
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
out_prefix = out_dir + '/' + runid + '-'
# info log file
infofh = logging.FileHandler(out_prefix + 'info.log')
infofh.setLevel(logging.INFO)
infofh.setFormatter(logging.Formatter(
        fmt='%(message)s',
))
logger.addHandler(infofh)
# verbose log file
vfh = logging.FileHandler(out_prefix + 'verbose.log')
vfh.setLevel(logging.DEBUG)
vfh.setFormatter(logging.Formatter(
        fmt='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
))
logger.addHandler(vfh)

env = dict()
# git
env['version'] = subprocess.check_output(["git", "describe"]).strip()
if not isinstance(env['version'], str):
    env['version'] = str(env['version'], "UTF-8")

# OMP
env['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', str())

# Data control
env['MAX_PAIRS'] = int(os.environ.get('MAX_PAIRS', 1000000))
env['BATCH_SIZE'] = int(os.environ.get('BATCH_SIZE', 10000))
env['EPOCHS'] = int(os.environ.get('EPOCHS', 10))
env['SPLIT'] = [int(el) for el in os.environ.get('SPLIT', '60,20,20').split(',')]

# Neural network
env['NN_UNITS'] = [int(el) for el in os.environ.get('NN_UNITS', '10').split(',')]
env['NN_LEARNING_RATE'] = float(os.environ.get('NN_LEARNING_RATE', '0.1'))

# Load model weights from file, don't train?
env['MODEL'] = os.environ.get('MODEL', None)

# Clustering
env['CLUSTER_SAMPLES'] = int(os.environ.get('CLUSTER_SAMPLES', 500))

# Perform tests if EPS and min_samples are set
try:
    env['TEST_EPS'] = float(os.environ['TEST_EPS'])
except:
    env['TEST_EPS'] = None
try:
    env['TEST_MS'] = int(os.environ['TEST_MS'])
except:
    env['TEST_MS'] = None

logger.info("Starting.")
logger.info("env: " + str(env))
for k in sorted(env.keys()):
    logger.info('env[\'{}\']: {}'.format(k,env[k]))


# In[ ]:

seed = 1470300368 # Unix time at time of writing
def rndseed():
    global seed
    seed += 1
    return seed


# ## Build network

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
l_lstm = l_emb
for i, num_units in enumerate(env['NN_UNITS']):
    logger.info('Adding {} units for {} layer'.format(num_units, i))
    l_lstm = LSTMLayer(l_lstm, num_units=num_units, name='LSTM-LAYER[{}]'.format(i), mask_input=l_mask)
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
updates = sgd(loss, params, learning_rate=env['NN_LEARNING_RATE'])

# Testing Procedure
test_prediction = get_output(cos_net, deterministic=True)
test_loss = binary_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(test_prediction > 0.5, target_var),
                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, input_var2, mask_var, mask_var2, target_var], loss, updates=updates)
val_fn = theano.function([input_var, input_var2, mask_var, mask_var2, target_var], [test_loss, test_acc])
prediction_fn = theano.function([input_var, input_var2, mask_var, mask_var2], prediction)
logger.debug("Spent {}s compilling.".format(time.time()-t))


# In[ ]:

alert_to_vector = theano.function([input_var, mask_var], get_output(l_slice))


# ## Load data

# In[ ]:

data = pd.read_csv('data/own-recordings/alerts-merged-cleaned.log.1465471791')

# Test data
test_incidents = np.array(['1', '2', 'benign']*2)
test_alerts = np.array(
    [
        'alert %s of incident %s' % (a, i)
         for i, a in zip(test_incidents, np.arange(len(test_incidents)))
    ]
)
test_data1 = pd.DataFrame(
    np.concatenate([test_incidents.reshape(6, 1), test_alerts.reshape(6, 1)], axis=1),
    columns=['incident', 'alert']
)
test_data1['cut'] = 0
test_data2 = test_data1.copy()[:2]
test_data2['cut'] = 1
test_data = pd.concat([test_data1, test_data2]).reset_index(drop=True)


# In[ ]:

n = 10
print('Limiting to %.3e samples' % n)
data = data_tmp.sample(n)
print('Expecting %.3e pairs' % ((n * np.array([0.6, 0.2, 0.2])) ** 2).sum())


# ## Encode

# In[ ]:

max_len = data['alert'].apply(len).max()

def encode_alert(alert):
    encoded_alert = np.zeros(max_len, dtype='int8')
    encoded_alert[:len(alert)] = map(ord, alert)
    return encoded_alert

def build_mask(encoded_alert):
    mask = (encoded_alert > 0).astype('int8')
    return mask

# Test
test_alert = data['alert'].iloc[0]
test_encoded_alert = encode_alert(test_alert)
test_mask = build_mask(test_encoded_alert)
assert ''.join(map(chr,test_encoded_alert[test_mask.astype(bool)])) == test_alert,    "First alert cannot be encoded and decoded: %s" % test_alert

data['encoded_alert'] = data['alert'].apply(encode_alert)
data['mask'] = data['encoded_alert'].apply(build_mask)

# incident as int, -1 represent benign
data['incident'] = pd.to_numeric(data['incident'], errors='coerce').fillna(-1).astype(int)


# ## Build pairs

# In[ ]:

# Test cartesian
test_pairs = pd.merge(test_data, test_data, on='cut')
assert len(test_pairs) ==     len(test_data[test_data['cut']==0])**2 +    len(test_data[test_data['cut']==1])**2
assert len(test_pairs[test_pairs['cut']==0]) == len(test_data[test_data['cut']==0])**2
assert len(test_pairs[test_pairs['cut']==1]) == len(test_data[test_data['cut']==1])**2

# Implement cartesian
pairs = pd.merge(data, data, on='cut')

# calculate correlation
def is_correlated(row):
    """
    row[0], row[1] : ints for incidents, with benign encoded as -1
    """
    return (row[0]!=-1) & (row[1]!=-1) & (row[0]==row[1])

pairs['cor'] = pairs[['incident_x', 'incident_y']].apply(is_correlated, raw=True, axis=1)
assert sum(data.groupby('cut').size()**2) == len(pairs)

print('%e' % len(pairs))


# ## Load model

# In[ ]:

if env['MODEL']:
    logger.info('Loading model from {}'.format(env['MODEL']))
    with open(env['MODEL']) as f:
        model = json.loads(f.read())

    # Order accoording to current model (JSON might reorder)
    params = get_all_params(cos_net)
    values = [np.array(model['model'][p.name]) for p in params]

    set_all_param_values(cos_net, values)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ## Train

# In[ ]:

if not env['MODEL']:
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
    logger.info('Training complete')


# ## Test

# In[ ]:

"""
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
"""


# ## Dump model

# In[ ]:

model_file = out_prefix + 'model.json'
model = {'model':{str(p): v.tolist() for p, v in zip(get_all_params(cos_net), get_all_param_values(cos_net))}}
logger.info('Saving model to {}'.format(model_file))
with open(model_file, 'w') as f:
    f.write(json.dumps(model))
logger.info('Model saved')


# ## Plot

# In[ ]:

# might, might not have x available
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
# might, might not be notebook
try:
    get_ipython().magic(u'matplotlib inline')
except NameError:
    pass

error_dict = dict()

land = np.logical_and
lnot = np.logical_not
    
for batch in iterate_minibatches(get_val_batch(), len(list(get_val_batch())), keep_incidents=True):
    alerts1, alerts2, masks1, masks2, corelations, iz, js = batch
    pred_floats = prediction_fn(alerts1, alerts2, masks1, masks2)

    logger.debug("Calculating error")
    positive = (pred_floats) > 0.5
    correct = np.equal(corelations, positive)
    true_positive = land(correct, positive)
    true_negative = land(correct, lnot(positive))
    false_positive = land(lnot(correct), positive)
    false_negative = land(lnot(correct), lnot(positive))

    logger.debug("Summing errors by incidents")
    for tp, tn, fp, fn, i, j in zip(
        true_positive, true_negative,
        false_positive, false_negative,
        iz, js
    ):
        error_dict[i] = error_dict.get(i, np.zeros(4)) + np.array([tp, tn, fp, fn])
        error_dict[j] = error_dict.get(j, np.zeros(4)) + np.array([tp, tn, fp, fn])
        
(labels, errors) = zip(*sorted(list(error_dict.items())))
errors = np.array(errors)
errors_norm = errors / errors.sum(axis=1)[:, None]



# In[ ]:

TP, TN, FP, FN = range(4)

error_count_latex = str()
error_count_latex += "\\hline\n" + " & ".join(['Incident', 'TP', 'TN', 'FP', 'FN']) + '\\\\\\hline \n \\hline \n'
for i, line in enumerate(errors):
    fmt = "{}" + " & ${:>8.0f} $"*4 + '\\\\\\hline \n'
    error_count_latex += fmt.format(i+1, *line)
logger.info('Error counts: \n' + error_count_latex)

# error rates
tpr = errors[:,TP] / (errors[:,TP] + errors[:,FN])
tnr = errors[:,TN] / (errors[:,TN] + errors[:,FP])
fpr = errors[:,FP] / (errors[:,TN] + errors[:,FP])
fnr = errors[:,FN] / (errors[:,TP] + errors[:,FN])
error_rates = np.vstack((tpr, tnr, fpr, fnr)).T

error_rates = np.vstack((error_rates, error_rates.mean(axis=0)[None,:])) # Average rates

error_rates_latex = str()
error_rates_latex += "\\hline\n" + " & ".join(['Incident', 'TPR', 'TNR', 'FPR', 'FNR']) + '\\\\\\hline \n \\hline \n'
for i, line in enumerate(error_rates):
    if i == 7:
        i = 'Avg.'
    else:
        i = i + 1
    fmt = ("{}" + " & ${:>8.2f} \%$"*4 + '\\\\\\hline \n')
    error_rates_latex += fmt.format(i, *line*100)
logger.info('Error rates: \n' + error_rates_latex)


# In[ ]:

import matplotlib.pyplot as plt

index = np.arange(len(labels))

fig, ax = plt.subplots()

bar_width = 0.2



types = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
colors = ['g', 'b', 'r', 'y']

for i, (typ, color) in enumerate(zip(types, colors)):
    rect = plt.bar(
        index + bar_width*i,
        errors_norm[:,i],
        bar_width,
        alpha=0.8,
        color=color,
        error_kw={'ecolor': '0.3'},
        label=typ,
    )

plt.xlabel('Incident')
plt.ylabel('Rate')
plt.xticks(index + bar_width, labels)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.tight_layout()

plt.savefig(out_prefix+'detection_norm.pdf', bbox_inches='tight')


# In[ ]:

for i, (typ, color) in enumerate(zip(types, colors)):
    rect = plt.bar(
        index + bar_width*i,
        errors[:,i],
        bar_width,
        alpha=0.8,
        color=color,
        error_kw={'ecolor': '0.3'},
        label=typ,
    )

plt.xlabel('Incident')
plt.ylabel('Count (Pairs)')
plt.xticks(index + bar_width, labels)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.tight_layout()

plt.savefig(out_prefix+'detection_notnorm.pdf', bbox_inches='tight')


# # Clustering

# In[ ]:

# data, encoded and structured for clustering

def _get_alerts(alerts):
    """
    Encode and structures single alerts similarly to pairs

    Similar to _get_batch, but ommitting second alerts and correlation.
    """
    alerts, masks, incidents = encode(alerts)
    if len(alerts) > env['CLUSTER_SAMPLES']:
        alerts = alerts[:env['CLUSTER_SAMPLES'],:]
        masks = masks[:env['CLUSTER_SAMPLES'],:]
        incidents = incidents[:env['CLUSTER_SAMPLES']]
    return alerts, masks, incidents

if env.get('CUT_NONE', False):
    get_train_alerts = get_val_alerts = get_test_alerts = lambda:_get_alerts(alerts)

elif env.get('CUT_INC', False):
    get_train_alerts = lambda:_get_alerts(alerts_train)
    get_val_alerts = lambda:_get_alerts(alerts_val)
    get_test_alerts = lambda:_get_alerts(alerts_test)

elif env.get('CUT_ALERT', False):
    get_train_alerts = lambda:_get_alerts(alerts_train)
    get_val_alerts = lambda:_get_alerts(alerts_val)
    get_test_alerts = lambda:_get_alerts(alerts_test)

elif env.get('CUT_PAIR', False):
    def _get_alerts(batch_it):
        """Get batch and drop unused vectors - for join and shuffle efffects"""
        alerts, _, masks, _, _, incidents, _ = zip(*batch_it())
        alerts = np.array(alerts)
        masks = np.array(masks)
        incidents = np.array(incidents)
        if len(alerts) > env['CLUSTER_SAMPLES']:
            alerts = alerts[:env['CLUSTER_SAMPLES'],:]
            masks = masks[:env['CLUSTER_SAMPLES'],:]
            incidents = incidents[:env['CLUSTER_SAMPLES']]
        return alerts, masks, incidents

    get_train_alerts = lambda:_get_alerts(get_train_batch)
    get_val_alerts = lambda:_get_alerts(get_val_batch)
    get_test_alerts = lambda:_get_alerts(get_test_batch)

else:
    raise NotImplementedError("No cut selected")


# ## Cluster train data

# In[ ]:

from sklearn.cluster import DBSCAN
from sklearn import metrics

logger.info("Clustering of alerts")

alerts_matrix, masks_matrix, incidents_vector = get_train_alerts()
X = alert_to_vector(alerts_matrix, masks_matrix)
y = incidents_vector
logger.info("Breakdown of labels:\n"+ break_down_data(y))

logger.info("Precomputing distances")
precomp_dist = np.zeros(shape=(len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        precomp_dist[i, j] = sp.spatial.distance.cosine(X[i], X[j])


# In[ ]:

logger.info("Running clustering algorithm")
epss = np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])
min_sampless = np.array([1, 3, 10, 30])
homogenity = np.zeros(shape=(len(epss), len(min_sampless)))
n_clusters = np.zeros_like(homogenity, dtype=int)
f1 = np.zeros_like(homogenity)
noise = np.zeros_like(homogenity, dtype=int)
cl_model = np.zeros_like(homogenity).tolist()
mapper = np.zeros_like(homogenity).tolist()

def build_cluster_to_incident_mapper(y, y_pred):
    # Assign label to clusters according which incident has the largest part of its alert in the given cluster
    # weight to handle class skew
    weights = {l: 1/cnt for (l, cnt) in zip(*np.unique(y, return_counts=True))}
    allocs = zip(y, y_pred)

    from collections import Counter
    c = Counter(map(tuple, allocs))

    mapper = dict()
    for _, (incident, cluster) in sorted([(c[k]*weights[k[0]], k) for k in c.keys()]):
        mapper[cluster] = incident

    mapper[-1] = -1 # Don't rely on what DBSCAN deems as noise
    return mapper

for i, eps in enumerate(epss):
    for j, min_samples in enumerate(min_sampless):
        cl_model[i][j] = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        y_pred = cl_model[i][j].labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters[i,j] = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        noise[i,j] = sum(y_pred == -1)
        homogenity[i,j] = metrics.homogeneity_score(y, y_pred)
        
        mapper[i][j] = build_cluster_to_incident_mapper(y, y_pred)
        y_pred_inc = np.array([mapper[i][j][el] for el in y_pred])
        f1[i,j] = metrics.f1_score(y, y_pred_inc, average='weighted')

        logger.info(
            "DBSCAN with (eps, min_samples)=({:1.0e},{:>2d}), n_clusters={:>3d}, homogenity={:1.3f}, f1={:1.3f}, noise={:>3d}".format(
                eps, min_samples, n_clusters[i,j], homogenity[i,j], f1[i,j], noise[i,j],
            ))


# In[ ]:

def param_plot_prepare(
    title,
):
    fig, ax = plt.subplots()
    ax.set_title(title)

    ax.set_yscale('log')
    ax.set_ylabel('min_samples')
    ax.set_yticklabels(min_sampless)
    ax.set_yticks(min_sampless)
    ax.set_ylim(min_sampless[0]/3, min_sampless[-1]*3)

    ax.set_xscale('log')
    ax.set_xlabel('eps')
    ax.set_xticklabels(epss)
    ax.set_xticks(epss)
    ax.set_xlim(epss[0]/3,epss[-1]*3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0e'))


def param_plot_scatter(
    data,
    xcoords,
    ycoords,

):
    # scaling
    data = np.copy(data)
    data = data-data.min() # Range to start at zero
    data = data/data.max() # Range to end at one

    coords = np.array([
            (i, j)
            for i in range(data.shape[0])
            for j in range(data.shape[1])
    ])

    plt.scatter(
        xcoords[coords[:,0]],
        ycoords[coords[:,1]],
        data[coords[:,0], coords[:,1]]*1000,
        c='white',
        alpha=0.5
    )


def param_plot_annotate(
    data,
    xcoords,
    ycoords,
    fmt='{}',
):
    coords = np.array([
            (i, j)
            for i in range(data.shape[0])
            for j in range(data.shape[1])
    ])
        
    for x, y, label in zip(
        xcoords[coords[:,0]],
        ycoords[coords[:,1]],
        data[coords[:,0], coords[:,1]],
    ):
        plt.annotate(
            fmt.format(label),
            xy = (x, y), xytext = (0, 0),
            textcoords = 'offset points', ha = 'center', va = 'center',
        )


def param_plot_save(filename):
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


# In[ ]:

param_plot_prepare('Cluster homogenity by DBSCAN parameters')
param_plot_scatter(homogenity, epss, min_sampless)
param_plot_annotate(homogenity, epss, min_sampless, fmt='{:.2f}')
param_plot_save(out_prefix+'cluster_homogenity.pdf')

param_plot_prepare('Cluster count by DBSCAN parameters')
param_plot_scatter(n_clusters, epss, min_sampless)
param_plot_annotate(n_clusters, epss, min_sampless)
param_plot_save(out_prefix+'cluster_count.pdf')

param_plot_prepare('Detection performance (F1) by DBSCAN parameters')
param_plot_scatter(f1, epss, min_sampless)
param_plot_annotate(f1, epss, min_sampless, fmt='{:.2f}')
param_plot_save(out_prefix+'cluster_detection.pdf')



# ## Cluster validation data

# In[ ]:

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
    
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        for i, x_core in enumerate(dbscan_model.components_): 
            if  metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


# In[ ]:

logger.info("Applying clusters to validation data")

alerts_matrix, masks_matrix, incidents_vector = get_val_alerts()
X = alert_to_vector(alerts_matrix, masks_matrix)
y = incidents_vector

logger.info("Breakdown of labels:\n"+ break_down_data(y))

for i, eps in enumerate(epss):
    for j, min_samples in enumerate(min_sampless):
        y_pred = dbscan_predict(cl_model[i][j], X)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters[i,j] = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        noise[i,j] = sum(y_pred == -1)
        homogenity[i,j] = metrics.homogeneity_score(y, y_pred)
        
        y_pred_inc = np.array([mapper[i][j][el] for el in y_pred])
        f1[i,j] = metrics.f1_score(y, y_pred_inc, average='weighted')

        logger.info(
            "Validation of DBSCAN with (eps, min_samples)=({:1.0e},{:>2d}), n_clusters={:>3d}, homogenity={:1.3f}, f1={:1.3f}, noise={:>3d}".format(
                eps, min_samples, n_clusters[i,j], homogenity[i,j], f1[i,j], noise[i,j],
            ))


# In[ ]:

param_plot_prepare('Validation of cluster homogenity by DBSCAN parameters')
param_plot_scatter(homogenity, epss, min_sampless)
param_plot_annotate(homogenity, epss, min_sampless, fmt='{:.2f}')
param_plot_save(out_prefix+'cluster_homogenity_val.pdf')

param_plot_prepare('Validation of cluster count by DBSCAN parameters')
param_plot_scatter(n_clusters, epss, min_sampless)
param_plot_annotate(n_clusters, epss, min_sampless)
param_plot_save(out_prefix+'cluster_count_val.pdf')

param_plot_prepare('Validation of detection performance (F1) by DBSCAN parameters')
param_plot_scatter(f1, epss, min_sampless)
param_plot_annotate(f1, epss, min_sampless, fmt='{:.2f}')
param_plot_save(out_prefix+'cluster_detection_val.pdf')



# ## Clustering - test data

# In[ ]:

if env['TEST_MS'] and env['TEST_EPS']:
    logger.info('Continuing to use test data (eps={}, min_samples={})'.format(
            env['TEST_MS'], env['TEST_EPS'],
    ))
else:
    logger.info('Validation results completed, exiting')
    sys.exit(0)


# In[ ]:

eps = env['TEST_EPS']
min_samples = env['TEST_MS']
i = epss.tolist().index(eps)
j = min_sampless.tolist().index(min_samples)

logger.info("Applying clusters to test data")

alerts_matrix, masks_matrix, incidents_vector = get_test_alerts()
X = alert_to_vector(alerts_matrix, masks_matrix)
y = incidents_vector

y_pred = dbscan_predict(cl_model[i][j], X)
y_pred_inc = np.array([mapper[i][j][el] for el in y_pred])


# In[ ]:

logger.info(
    "Incident(i) to cluster(j) \"confusion matrix\":\n"+
    str(metrics.confusion_matrix(y, y_pred))
)

labels = ['noise'] + range(n_clusters[i][j])
desired_rows = range(1,8)

res = "\\hline\n & " + " & ".join([str(c) for c in labels]) + '\\\\\\hline\n\\hline\n'
for l, row in zip(labels, metrics.confusion_matrix(y, y_pred)):
    if l not in desired_rows:
        continue # No incident zero or noise as row
    fmt = ('{}' + " & {} "*len(row))
    res += fmt.format(l, *row) + '\\\\\\hline\n'
logger.info(res)



# In[ ]:

logger.info(
    "Incident(i) to incident(j) confusion matrix:\n"+
    str(metrics.confusion_matrix(y, y_pred_inc))
)

labels = ['noise'] + range(1,len(set(y))+1)
desired_rows = range(1,len(set(y))+1)

res = "\\hline\n & " + " & ".join([str(c) for c in labels]) + '\\\\\\hline\n\\hline\n'
for l, row in zip(labels, metrics.confusion_matrix(y, y_pred_inc)):
    if l not in desired_rows:
        continue 
    fmt = ('{}' + " & {} "*len(row))
    res += fmt.format(l, *row) + '\\\\\\hline\n'
logger.info(res)


# In[ ]:

cm = metrics.confusion_matrix(y, y_pred_inc)
logger.info("Correctly classified, disregarding noise: {:.2f}%".format(cm[1:,1:].diagonal().sum()/cm[1:,1:].sum()*100))
logger.info("Correctly classified, including noise: {:.2f}%".format(cm[1:,1:].diagonal().sum()/cm.sum()*100))
logger.info("Noise sample count: {}".format(sum(y_pred_inc == -1)))


# In[ ]:

logger.info(
    "Classification report:\n"+
    metrics.classification_report(y, y_pred_inc)
)


# ## Analysing results

# In[ ]:

def uniq_counts(l, sort_key=itemgetter(1), sort_reverse=True):
    try:
        l = [json.dumps(el) for el in l]
        logger.debug('dumped to json')
    except:
        l = [str(el) for el in l]
        logger.warn('failed to dump to json, casting to str')
    unique, unique_counts = np.unique(l, return_counts=True)
    res = sorted(
        zip(unique, unique_counts),
        key=sort_key,
        reverse=sort_reverse,
    )
    unique, count = map(list, zip(*res))
    try:
        unique = list(map(json.loads, unique))
        logger.debug('read from json')
    except:
        logger.warn('failed to read as json, returning as str')
    return unique, count

# a bogus incident to hold all the unclassifiable test alerts
noise_alerts = alerts_matrix[y_pred_inc == -1]
noise_masks = masks_matrix[y_pred_inc == -1]

def decode(alert, mask):
    alert = alert[mask.astype(bool)] # apply mask
    alert = [chr(c) for c in alert]
    return ''.join(alert)

noise_incident = (None, [decode(a, m) for a, m in zip(noise_alerts, noise_masks)])


# In[ ]:

logger.info('Counts of different noise alerts')
uniq, counts = uniq_counts(
    pool(modify(
            [noise_incident],
            [mask_ips, mask_tss, mask_ports],
)))
for i, (u, c) in enumerate(zip(uniq, counts)):
    logger.info("{:2d}. n={:2d}: {}".format(i+1, c, u[1]))


# In[ ]:

logger.info('Priority among noise alerts:')
uniq, counts = uniq_counts(pool(modify(
            [noise_incident],
            [extract_prio],
)), sort_key=itemgetter(0), sort_reverse=False)
for u, c in zip(uniq, counts):
    logger.info("Priority {}, n={:3d}, {:5.2f}%".format(u[1], c, c/sum(counts)*100))


# In[ ]:

logger.info('Priority of all alerts:')
all_alerts = pool(modify(
        incidents,
        [extract_prio],
))
all_alerts = [a[1] for a in all_alerts] # discard incident id
uniq, counts = uniq_counts(
    all_alerts, 
    sort_key=itemgetter(0), sort_reverse=False,
)
for u, c in zip(uniq, counts):
    logger.info("Priority {}, n={:4d}, {:5.2f}%".format(u, c, c/sum(counts)*100))


# In[ ]:

logger.info('Priority of all alerts by incident:')
uniq, counts = uniq_counts(pool(modify(
        incidents,
        [extract_prio],
    )),
    sort_key=itemgetter(0), sort_reverse=False,
)
uniq = np.array(uniq)
counts = np.array(counts)
cnt_array = np.zeros([uniq[:,0].max(), uniq[:,1].max()], dtype=int)
uniq = uniq-1 # incident and prio: 1-indexed, arrays: 0-indexed
for (i, p), c in zip(uniq, counts):
    cnt_array[i,p] = c

s = "Counts:\n"
fmt = '{:<11}' + '{:>8}'*cnt_array.shape[1] + '\n'
s += fmt.format('', 'Prio 1', 'Prio 2', 'Prio 3')
for i in np.arange(cnt_array.shape[0]):
    s += fmt.format(*['Incident '+str(i+1)]+list(cnt_array[i,:]))
logger.info(s)

s = "Normalised pr. incident:\n"
fmt = '{:<11}' + '{:>8}'*cnt_array.shape[1] + '\n'
s += fmt.format('', 'Prio 1', 'Prio 2', 'Prio 3')
fmt = '{:<11}' + '{:7.2f}%'*cnt_array.shape[1] + '\n'
for i in np.arange(cnt_array.shape[0]):
    s += fmt.format(*['Incident '+str(i+1)]+list((100*cnt_array[i,:]/cnt_array.sum(axis=1)[i])))
logger.info(s)

s = "Normalised across incident:\n"
fmt = '{:<11}' + '{:>8}'*cnt_array.shape[1] + '\n'
s += fmt.format('', 'Prio 1', 'Prio 2', 'Prio 3')
fmt = '{:<11}' + '{:7.2f}%'*cnt_array.shape[1] + '\n'
for i in np.arange(cnt_array.shape[0]):
    s += fmt.format(*['Incident '+str(i+1)]+list(100*cnt_array[i,:]/cnt_array.sum()))
logger.info(s)


# In[ ]:



