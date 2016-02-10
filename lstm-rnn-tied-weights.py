
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
from lstm_rnn_tied_weights import load, modify, split, pool, cross_join, limit
from lstm_rnn_tied_weights import iterate_minibatches
from lstm_rnn_tied_weights import mask_ips, mask_tss, uniquify_victim
logger = lstm_rnn_tied_weights.logger

env = dict()
# git
env['version'] = subprocess.check_output(["git", "describe"]).strip()
if not isinstance(env['version'], str):
    env['version'] = str(env['version'], "UTF-8")

# OMP
env['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', str())

# Masking/modifying
env['MASKING'] = os.environ.get('MASKING', str())
env['MASK_IP'] = 'ip' in env['MASKING'].lower()
env['MASK_TS'] = 'ts' in env['MASKING'].lower()
env['UNIQUIFY_VICTIM'] = 'true' in os.environ.get('UNIQUIFY_VICTIM', 'true').lower()

# cutting
env['CUT'] = os.environ.get('CUT', str())
if 'none' in env['CUT'].lower():
    env['CUT_NONE'] = True
elif 'inc' in env['CUT'].lower():
    env['CUT_INC'] = True
elif 'alert' in env['CUT'].lower():
    env['CUT_ALERT'] = True
elif 'pair' in env['CUT'].lower():
    env['CUT_PAIR'] = True
else:
    raise NotImplementedError("Pleas set CUT={none|inc|alert|pair} (CUT={})".format(env['CUT']))

# Data control
env['MAX_PAIRS'] = int(os.environ.get('MAX_PAIRS', 1000000))
env['BATCH_SIZE'] = int(os.environ.get('BATCH_SIZE', 10000))
env['EPOCHS'] = int(os.environ.get('EPOCHS', 10))
env['SPLIT'] = [int(el) for el in os.environ.get('SPLIT', '60,20,20').split(',')]

# Metadata
env['VICTIM_IP'] = '147.32.84.165'

# Neural network
env['NN_UNITS'] = [int(el) for el in os.environ.get('NN_UNITS', '10').split(',')]
env['NN_LEARNING_RATE'] = float(os.environ.get('NN_LEARNING_RATE', '0.1'))

logger.info("Starting.")
logger.info("env: " + str(env))
for k in sorted(env.keys()):
    logger.info('env[\'{}\']: {}'.format(k,env[k]))


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


# ## Prepare data

# In[ ]:

def break_down_data(labels):
    u, c = np.unique(labels, return_counts=True)
    c_norm = (c/sum(c))*100
    result = str()
    result += 'Label: '+('{: >10}'*len(u)).format(*u) + '\n'
    result += 'Count: '+('{: >10}'*len(u)).format(*c) + '\n'
    result += 'Norm.: '+('{: >9.2f}%'*len(u)).format(*c_norm)
    return result

# If/what to mask out or modify
modifier_fns = []
if env['MASK_IP']:
    modifier_fns.append(mask_ips)
if env['MASK_TS']:
    modifier_fns.append(mask_tss)
if env['UNIQUIFY_VICTIM']:
    modifier_fns.append(lambda incidents: uniquify_victim(incidents, env['VICTIM_IP']))

def _get_batch(
        alerts,
        max_pairs,
        offset=0,
):
    for sample in limit(
            cross_join(alerts, offset=offset),
            max_pairs,
    ):
        yield sample

incidents = load([
    'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-140-2/2015-10-27_capture-win11.pcap.shifted.out',
#    'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-141-1/2015-23-10_win10.pcap.shifted.out',
#    'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-69/2014-04-07_capture-win17.pcap.shifted.out',
#    'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/botnet-capture-20110811-neris.pcap.shifted.out',
])
incidents = modify(incidents, modifier_fns)
alerts = pool(incidents)

pair_cnt = min([env['MAX_PAIRS'], len(alerts)**2])
logger.info('Maximum possible pairs; pair_cnt={} env[\'MAX_PAIRS\']={}, len(alerts)**2={}'.format(
        pair_cnt,
        env['MAX_PAIRS'],
        len(alerts)**2,
    ))
maxes = (np.array(env['SPLIT'])/sum(env['SPLIT'])*pair_cnt).astype(int)
logger.info('train, val and test max pairs: {}'.format(maxes))
train_max, val_max, test_max = maxes
logger.info('Breakdown of original data:\n'+break_down_data([i[0] for i in pool(incidents)])+'\n')

if env.get('CUT_NONE', False):
    get_train_batch = lambda: _get_batch(alerts, train_max)
    get_val_batch = lambda: _get_batch(alerts, val_max)
    get_test_batch = lambda: _get_batch(alerts, test_max)

elif env.get('CUT_INC', False):
    incidents = split(incidents, env['SPLIT'])
    alerts_train, alerts_val, alerts_test = tuple(map(pool, incidents))

    get_train_batch = lambda: _get_batch(alerts_train, train_max)
    get_val_batch = lambda: _get_batch(alerts_val, val_max)
    get_test_batch = lambda: _get_batch(alerts_test, test_max)

elif env.get('CUT_ALERT', False):
    alerts_train, alerts_val, alerts_test = split(alerts, env['SPLIT'])

    get_train_batch = lambda: _get_batch(alerts_train, train_max)
    get_val_batch = lambda: _get_batch(alerts_val, val_max)
    get_test_batch = lambda: _get_batch(alerts_test, test_max)

elif env.get('CUT_PAIR', False):
    get_train_batch = lambda: _get_batch(alerts, train_max, offset=0)
    get_val_batch = lambda: _get_batch(alerts, val_max, offset=train_max)
    get_test_batch = lambda: _get_batch(alerts, test_max, offset=train_max+val_max)

else:
    raise NotImplementedError("No cut selected")


# ## Load model

# logger.debug('Loading model')
# with open('model.log') as f:
#     model = json.loads(f.read())
# 
# keys = get_all_params(cos_net)
# keys = [str(k).replace('[0]','') for k in keys]
# 
# params = [np.array(model['model'][k]) for k in keys]
# set_all_param_values(cos_net, params)
# 

# ## Train

# In[ ]:

a1, a2, m1, m2, cor, inc1, inc2 = range(7)
for cut, batch_fn in [
    ('training', get_train_batch),
    ('validation', get_val_batch),
    ('testing', get_test_batch),
]:
    logger.info('Breakdown of {} data;\n'.format(cut))
    logger.info('correlation:\n'+break_down_data([p[cor] for p in batch_fn()]))
    logger.info('incident 1:\n'+break_down_data([p[inc1] for p in batch_fn()]))
    logger.info('incident 2:\n'+break_down_data([p[inc2] for p in batch_fn()])+'\n')

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


# ## Test

# In[ ]:

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


# ## Plot

# In[ ]:

get_ipython().magic(u'matplotlib inline')

error_dict = dict()

land = np.logical_and
lnot = np.logical_not
    
for batch in iterate_minibatches(get_test_batch(), test_max, keep_incidents=True):
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

import matplotlib.pyplot as plt

index = np.arange(len(labels))

fig, ax = plt.subplots()

bar_width = 0.1



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

plt.xlabel('Incidents')
plt.ylabel('Rate')
plt.xticks(index + bar_width, labels)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.tight_layout()

plt.savefig('detection_norm.pdf', bbox_inches='tight')


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

plt.xlabel('Incidents')
plt.ylabel('Rate')
plt.xticks(index + bar_width, labels)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.tight_layout()

plt.savefig('detection_notnorm.pdf', bbox_inches='tight')


# In[ ]:

sys.exit(0)


# ## Clustering

# In[ ]:

X = alert_to_vector(alerts1, masks1)
y = iz


# In[ ]:

from sklearn.cluster import DBSCAN

def my_cluster_eval(y, y_pred, X, n_clusters):
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, y_pred))
    print("Completeness: %0.3f" % metrics.completeness_score(y, y_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(y, y_pred))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(y, y_pred))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(y, y_pred))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, y_pred))
    
for eps in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
    for min_samples in [1, 3, 10, 30]:
        db = DBSCAN(eps=0.01, min_samples=3).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        y_pred = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        
        logger.info(
            "DBSCAN with eps={} and min_samples={} yielded {} clusters with a homogenity of {}".format(
                eps, min_samples, n_clusters, metrics.homogeneity_score(y, y_pred)
            ))


# In[ ]:





my_cluster_eval(y, y_pred, X, n_clusters_)


# In[ ]:

np.unique(y_pred, return_counts=True)


# In[ ]:

# Assign label to clusters according which incident has the largest part of its alert in the given cluster
# weight to handle class skew
weights = {l: 1/cnt for (l, cnt) in zip(*np.unique(y, return_counts=True))}
allocs = zip(y, y_pred)

from collections import Counter
c = Counter(map(tuple, allocs))

mapper = dict()
for _, (incident, cluster) in sorted([(c[k]*weights[k[0]], k) for k in c.keys()]):
    mapper[cluster] = incident


# In[ ]:

# misclassification matrix
y_pred_inc = np.array([mapper[el] for el in y_pred])
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred_inc)

cm


# In[ ]:

(((cm+1.0e-16) / cm.sum(axis=0))*100)


# In[ ]:

# Map any cluster to the incident that it is most often put in

mapping = sorted(mapping)


# In[ ]:

mapper = dict()
for _, m in mapping:
    mapper[m[0]] = m[1]
    
y_pred = np.array([mapper[el] for el in y_pred])

print(mapper.keys())
print(mapper.values())


# In[ ]:




# In[ ]:

weights = {l: 1/cnt for (l, cnt) in zip(*np.unique(y, return_counts=True))}


# In[ ]:

zip(*np.unique(y, return_counts=True))


# In[ ]:



