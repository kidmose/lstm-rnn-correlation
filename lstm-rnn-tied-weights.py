
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
import psutil
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
import pandas as pd

import matplotlib
try: # If X is not available select backend not requiring X
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
try: # If ipython, do inline
    get_ipython().magic(u'matplotlib inline')
except NameError:
    pass
import matplotlib.pyplot as plt

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


# In[ ]:

logger = lstm_rnn_tied_weights.logger
OUTPUT = 'output'
p_start = psutil.Process(os.getpid()).create_time()
p_start = time.localtime(p_start)
p_start = datetime.datetime.fromtimestamp(time.mktime(p_start))
runid = p_start.strftime("%Y%m%d-%H%M%S-") + socket.gethostname()
if os.environ.get('SLURM_JOB_ID', False):
    runid += '-slurm-' + os.environ['SLURM_JOB_ID']
out_dir = OUTPUT + '/' + runid
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

logger.info('Output, including logs, are going to: {}'.format(out_dir))


# In[ ]:

env = dict()

# Data control
env['MAX_PAIRS'] = int(os.environ.get('MAX_PAIRS', 0))
env['BATCH_SIZE'] = int(os.environ.get('BATCH_SIZE', 10000))
env['RAND_SEED'] = int(os.environ.get('RAND_SEED', time.time())) # Current unix time if not specified
env['EPOCHS'] = int(os.environ.get('EPOCHS', 10))

# Neural network
env['NN_UNITS'] = [int(el) for el in os.environ.get('NN_UNITS', '10').split(',')]
env['NN_LEARNING_RATE'] = float(os.environ.get('NN_LEARNING_RATE', '0.1'))

# git
env['version'] = subprocess.check_output(["git", "describe"]).strip()
if not isinstance(env['version'], str):
    env['version'] = str(env['version'], "UTF-8")

# Platform
env['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', str())
env['THEANO_FLAGS'] = os.environ.get('THEANO_FLAGS', str())
env['MAX_HOURS'] = float(os.environ.get('MAX_HOURS', '23.5'))
start_script = datetime.datetime.now()
end_script_before = start_script + datetime.timedelta(hours=env['MAX_HOURS'])
logger.info('Started at {}, must end before {}'.format(start_script, end_script_before))

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

# Continue existing, old job
env['OLD_JOB'] = os.environ.get('OLD_JOB', None)
if env['OLD_JOB']:
    logger.critical("Continuing on OLD_JOB={}".format(env['OLD_JOB']))
    if not os.path.exists(env['OLD_JOB']):
        logger.critical("Old job to continue does not exist ({})".format(env['OLD_JOB']))
        sys.exit(-1)
    old_run_id = env['OLD_JOB'].split('/')[-1]
    old_out_prefix = env['OLD_JOB'] + '/' + old_run_id + '-'
    with open(old_out_prefix + 'env.json') as f:
        env_old = json.load(f)
    logger.info('Loaded old env')
    statics = [
        'MAX_PAIRS',
        'BATCH_SIZE',
        'NN_UNITS',
        'NN_LEARNING_RATE',
    ]
    for s in statics:
        env[s] = env_old[s]

logger.info("Starting.")
logger.info("env: " + str(env))
for k in sorted(env.keys()):
    logger.info('env[\'{}\']: {}'.format(k,env[k]))

logger.debug('Saving env')
with open(out_prefix + 'env.json', 'w') as f:
    json.dump(env, f)


# In[ ]:

seed = env['RAND_SEED']
def rndseed():
    global seed
    seed += 1
    return seed


# In[ ]:

class Timer(object):
    def __init__(self, name='', log=logger.debug):
        self.name = name
        self.log = log
        
    def __enter__(self):
        self.log('Timer(%s) started' % (self.name, ))
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.dur = datetime.timedelta(seconds=self.end - self.start)
        self.log('Timer(%s):\t%s' % (self.name, self.dur))


### Build network

# In[ ]:

# Data for unit testing
X_unit = ['abcdef', 'abcdef', 'qwerty']
X_unit = [[ord(c) for c in w] for w in X_unit]
X_unit = np.array(X_unit, dtype='int32')
logger.debug(X_unit)
n_alerts_unit, l_alerts_unit = X_unit.shape
mask_unit = np.ones(X_unit.shape, dtype=theano.config.floatX)
logger.debug(mask_unit)


# In[ ]:

# Dimensions
n_alerts = None
l_alerts = None
n_alphabet = 2**7 # All ASCII chars


# In[ ]:

# Symbolic variables
input_var = T.imatrix('inputs')
input_var2 = T.imatrix('inputs2')
mask_var = T.matrix('masks')
mask_var2 = T.matrix('masks2')
target_var = T.vector('targets')


#### First line

# In[ ]:

l_in = InputLayer(shape=(n_alerts, l_alerts), input_var=input_var, name='INPUT-LAYER')

l_in_output_var = get_output(l_in, inputs={l_in: input_var})
assert l_in_output_var.dtype == 'int32'

pred_unit = l_in_output_var.eval({input_var: X_unit})
assert (pred_unit == X_unit).all(), "Unexpected output"


# In[ ]:

l_emb = EmbeddingLayer(
    l_in, n_alphabet, n_alphabet,
    W=np.eye(n_alphabet, dtype=theano.config.floatX),
    name='EMBEDDING-LAYER',
)
l_emb.params[l_emb.W].remove('trainable') # Fix weight

l_emb_output_var = get_output(l_emb, inputs={l_in: input_var})
assert l_emb_output_var.dtype == theano.config.floatX

pred_unit = l_emb_output_var.eval({input_var: X_unit})
assert (np.argmax(pred_unit, axis=2) == X_unit).all()
assert np.all(pred_unit.shape == (n_alerts_unit, l_alerts_unit, n_alphabet ))


# In[ ]:

l_mask = InputLayer(shape=(n_alerts, l_alerts), input_var=mask_var, name='MASK-INPUT-LAYER')

l_mask_output_var = get_output(l_mask, inputs={l_mask: mask_var})
assert l_mask_output_var.dtype == theano.config.floatX

pred_unit = l_mask_output_var.eval({mask_var: mask_unit})
assert (pred_unit == mask_unit).all(), "Unexpected output"


# In[ ]:

l_lstm = l_emb
for i, num_units in enumerate(env['NN_UNITS']):
    logger.info('Adding {} units for {} layer'.format(num_units, i))
    l_lstm = LSTMLayer(l_lstm, num_units=num_units, name='LSTM-LAYER[{}]'.format(i), mask_input=l_mask)

l_lstm_output_var = get_output(l_lstm, inputs={l_in: input_var, l_mask: mask_var})
assert l_mask_output_var.dtype == theano.config.floatX

pred_unit = l_lstm_output_var.eval({input_var: X_unit, mask_var: mask_unit})
assert pred_unit.dtype == theano.config.floatX, "Unexpected dtype"
assert pred_unit.shape == (n_alerts_unit, l_alerts_unit, num_units), "Unexpected dimensions"
pred_unit = l_lstm_output_var.eval({input_var: [[1],[1]], mask_var: [[1],[1]]})
assert np.all(pred_unit[0] == pred_unit[1]), "Repeated alerts must produce the same"
pred_unit = l_lstm_output_var.eval({input_var: [[1,1],[1,1]], mask_var: [[1,1],[1,1]]})
assert np.all(pred_unit[0] == pred_unit[1]), "Repeated alerts must produce the same"
pred_unit = l_lstm_output_var.eval({input_var: [[1,1],[0,1]], mask_var: [[1,1],[1,1]]})
assert np.all(pred_unit[0] != pred_unit[1]), "Earlier must affect laters"
pred_unit = l_lstm_output_var.eval({input_var: [[1,0],[1,1]], mask_var: [[1,1],[1,1]]})
assert np.all(pred_unit[0,0] == pred_unit[1,0]), "Later must not affect earlier"
assert np.all(pred_unit[0,1] != pred_unit[1,1]), "Current must make a difference"


# In[ ]:

l_slice = SliceLayer(l_lstm, indices=-1, axis=1, name="SLICE-LAYER") # Only last timestep

l_slice_output_var = get_output(l_slice, inputs={l_in: input_var, l_mask: mask_var})
assert l_slice_output_var.dtype == theano.config.floatX

pred_unit = l_slice_output_var.eval({input_var: X_unit, mask_var: mask_unit})
assert pred_unit.shape == (n_alerts_unit, num_units), "Unexpected shape"
pred_unit_lstm = l_lstm_output_var.eval({input_var: X_unit, mask_var: mask_unit})
assert np.all(pred_unit_lstm[:, -1, :] == pred_unit), "Unexpected result of slicing"

net = l_slice
logger.info('First line built')


#### Second line as a copy with shared weights

# In[ ]:

l_in2 = InputLayer(shape=l_in.shape, input_var=input_var2, name=l_in.name+'2')
l_mask2 = InputLayer(shape=l_mask.shape, input_var=mask_var2, name=l_mask.name+'2')
net2 = lstm_rnn_tied_weights.clone(net, l_in2, l_mask2)

net_pred = get_output(net, inputs={l_in: input_var, l_mask: mask_var}).eval({input_var: X_unit, mask_var: mask_unit})
net_pred2 = get_output(net2, inputs={l_in2: input_var, l_mask2: mask_var}).eval({input_var: X_unit, mask_var: mask_unit})
assert (net_pred == net_pred2).all(), "Output mismatch, two lines must produce same output"

logger.info('Second line built')


#### Merge lines

# In[ ]:

l_cos = CosineSimilarityLayer(net, net2, name="COSINE-SIMILARITY-LAYER")

l_cos_output_var = get_output(l_cos, inputs={
        l_in: input_var,
        l_mask: mask_var,
        l_in2: input_var2,
        l_mask2: mask_var2,
})
assert l_emb_output_var.dtype == theano.config.floatX

pred_unit = l_cos_output_var.eval(({
            input_var: X_unit,
            input_var2: X_unit,
            mask_var: mask_unit,
            mask_var2: mask_unit,
}))


# In[ ]:

l_sig = NonlinearityLayer(l_cos, nonlinearity=sigmoid, name="SIGMOID-LAYER")

l_sig_output_var = get_output(l_sig, inputs={
        l_in: input_var,
        l_mask: mask_var,
        l_in2: input_var2,
        l_mask2: mask_var2,
})
assert l_sig_output_var.dtype == theano.config.floatX

pred_unit = l_sig_output_var.eval(({
            input_var: X_unit,
            input_var2: X_unit,
            mask_var: mask_unit,
            mask_var2: mask_unit,
}))
assert pred_unit is not None

cos_net = l_sig


# In[ ]:

with Timer('Compiling theano', logger.info):
    # Training Procedure
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

    alert_to_vector = theano.function([input_var, mask_var], get_output(l_slice))


### Load data

# In[ ]:

with Timer('Load data'):
    data = pd.read_csv('data/own-recordings/alerts-merged-cleaned-strat50.log.1465471791')

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


### Encode

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

with Timer('Encode alerts'):
    data['encoded_alert'] = data['alert'].apply(encode_alert)
    data['mask'] = data['encoded_alert'].apply(build_mask)

    # incident as int, -1 represent benign
    data['incident'] = pd.to_numeric(data['incident'], errors='coerce').fillna(-1).astype(int)

assert (data['alert'].map(len) == data['mask'].map(sum)).all(), "Sum of mask is not equal length of alert"
assert (data['mask'].map(np.nonzero).map(np.max)+1 == data['alert'].map(len)).all(),     "Last non-zero index of mask is not equal to length of alert"


### Cut data, pairing

# In[ ]:

def get_pairs(data):
    KEY = 'DUMMY_MERGE_KEY'
    assert KEY not in data.columns
    data = data.copy(deep=False)
    data[KEY] = 0
    pairs = pd.merge(data, data, on=KEY)
    pairs.drop(KEY , axis=1, inplace=True)
    return pairs

# Test get_pairs
test_pairs = get_pairs(test_data)
assert len(test_pairs) == len(test_data) ** 2

def is_correlated(row):
    """
    row[0], row[1] : ints for incidents, with benign encoded as -1
    """
    return (row[0]!=-1) & (row[1]!=-1) & (row[0]==row[1])

assert is_correlated([0, 0]), "Same incident must result in true"
assert not is_correlated([0, 1]), "Different incident must result in false"
assert not is_correlated([0, -1]), "Benign must result in false"
assert not is_correlated([-1, -1]), "Benign must result in false"

def add_cor_col(pairs):
    pairs = pairs.copy(deep=False)
    pairs['cor'] = pairs[['incident_x', 'incident_y']].apply(is_correlated, raw=True, axis=1)
    return pairs
    
def shuffle(pairs):
    np.random.seed(rndseed())
    return pairs.reindex(np.random.permutation(pairs.index))

def take_and_modify_cut(data, cut):
    data = data[(data['cut']==cut)]
    pairs = get_pairs(data)
    pairs = shuffle(pairs)
    pairs = add_cor_col(pairs)
    return pairs

with Timer('Build pairs'):
    pairs_train = take_and_modify_cut(data, 0)
    pairs_val = take_and_modify_cut(data, 1)
    pairs_test = take_and_modify_cut(data, 2)


# In[ ]:

def iterate_minibatches(pairs, batch_size, max_pairs=0, include_incidents=False):
    ii = 0 # minibatch counter
    if max_pairs != 0:
        logger.debug('Limiting to {}'.format(max_pairs))
        pairs = pairs.head(max_pairs)
    assert len(pairs) >= batch_size,         "{} samples is not enough to produce a minibatch of {} samples"        .format(len(pairs), batch_size)
    logger.info('Expect %d minibatches' % (len(pairs)//batch_size))
    while len(pairs) - ii * batch_size >= batch_size:
        with Timer('Minibatch{}'.format(ii)):
            begin = ii * batch_size
            ii += 1
            end = ii * batch_size
            logger.debug("Producing minibatch no. %d" % ii)
            batch = pairs.iloc[begin:end]
            inputs1 = np.array(batch['encoded_alert_x'].values.tolist())
            inputs2 = np.array(batch['encoded_alert_y'].values.tolist())
            masks1 = np.array(batch['mask_x'].values.tolist())
            masks2 = np.array(batch['mask_y'].values.tolist())
            targets = np.array(batch['cor'].values.tolist())
            if include_incidents:
                incidents1 = np.array(batch['incident_x'].values.tolist())
                incidents2 = np.array(batch['incident_y'].values.tolist())
                yield inputs1, inputs2, masks1, masks2, targets, incidents1, incidents2
            else:
                 yield inputs1, inputs2, masks1, masks2, targets


### Plot Empirical Distribution Functions for model output, by ground truth for correlation

# In[ ]:

pairs_train_edf_cor = pairs_train[pairs_train['cor']==True].head(1000)
pairs_train_edf_uncor = pairs_train[pairs_train['cor']==False].head(1000)
pairs_val_edf_cor = pairs_val[pairs_val['cor']==True].head(1000)
pairs_val_edf_uncor = pairs_val[pairs_val['cor']==False].head(1000)

_, edf_bins = np.histogram([], bins=100, range=(0,1))
edf_bin_centers = np.vstack((edf_bins[:-1].T, edf_bins[1:].T)).mean(axis=0)

def get_hist(pairs):
    inputs1, inputs2, masks1, masks2, targets = next(
        iterate_minibatches(pairs, batch_size=len(pairs)),
    )
    hist, _ = np.histogram(
        prediction_fn(inputs1, inputs2, masks1, masks2),
        bins=edf_bins,
    )
    return hist

def get_hists():
    return {
        'training' : {
            'correlated' : get_hist(pairs_train_edf_cor),
            'uncorrelated' : get_hist(pairs_train_edf_uncor),
        },
        'validation' : {
            'correlated' : get_hist(pairs_val_edf_cor),
            'uncorrelated' : get_hist(pairs_val_edf_uncor),
        },
    }

def plot_hists(hists):
    epochs = sorted(hists.keys())
    
    for sett in {'training', 'validation'}:
        for epoch in epochs:
            plt.figure()
            for c in {'correlated', 'uncorrelated'}:
                plt.plot(
                    edf_bin_centers,
                    hists[epoch][sett][c],
                    label=c,
                )
            plt.title(
                'EDF on %s data, trained for %d epochs' % (sett, epoch),
            )
            plt.legend()
            max_val = 0
            for v1 in hists.values():
                for v2 in v1.values():
                    for v3 in v2.values():
                        max_val = max(v3.max(), max_val)
            plt.gca().set_ylim(0, max_val)
            plt.savefig(
                out_prefix + 'edf_%s_%.2d.pdf' % (sett, epoch),
                bbox_inches='tight',
            )
            plt.close()


### Performance evaluation

# In[ ]:

def perf_eval():
    logger.debug('Starting performance evaluation on training data')
    train_err = 0
    train_acc = 0
    train_mbatches = 0
    for mbatch in iterate_minibatches(pairs_train, env['BATCH_SIZE'], env['MAX_PAIRS']):
        err, acc = val_fn(*mbatch)
        train_err += err
        train_acc += acc
        train_mbatches += 1
    train_err = train_err/train_mbatches
    train_acc = train_acc/train_mbatches
    logger.debug(
        'Completed performance evaluation on training data, err={}, acc={}'.format(
            train_err, train_acc,
        )
    )

    logger.debug('Starting performance evaluation on validation data')
    val_err = 0
    val_acc = 0
    val_mbatches = 0
    for mbatch in iterate_minibatches(pairs_val, env['BATCH_SIZE'], env['MAX_PAIRS']):
        err, acc = val_fn(*mbatch)
        val_err += err
        val_acc += acc
        val_mbatches += 1
    val_err = val_err/val_mbatches
    val_acc = val_acc/val_mbatches
    logger.debug(
        'Completed performance evaluation on validation data, err={}, acc={}'.format(
            val_err, val_acc,
        )
    )
    return {
        'training':{
            'error': train_err,
            'accuracy': train_acc,
        },
        'validation':{
            'error': val_err,
            'accuracy': val_acc,
        }
    }

def plot_perfs(perfs):
    for metric in ['accuracy', 'error']:
        plt.figure()
        for zet in ['training', 'validation']:
            y = [perf[zet][metric] for epoch, perf in sorted(perfs.items())]
            x = range(len(y))
            plt.plot(x, y, '.-', label='{}'.format(zet))
        plt.title('Performance ({}) over epochs'.format(metric))
        plt.legend()
        plt.xticks(perfs.keys())
        plt.savefig(
            out_prefix + 'perf_{}.pdf'.format(metric),
            bbox_inches='tight',
        )
        plt.close()


### Load model

# In[ ]:

def load_model(net, filename):
    logger.info('Loading model from {}'.format(filename))
    with open(filename) as f:
        model = json.loads(f.read())

    # Order according to current model (JSON might reorder)
    params = get_all_params(cos_net)
    values = [np.array(model['model'][p.name], dtype=theano.config.floatX) for p in params]

    set_all_param_values(cos_net, values)
    logger.info("Model loaded")

def dump_model(net, filename):
    model = {'model':{str(p): v.tolist() for p, v in zip(get_all_params(net), get_all_param_values(net))}}
    logger.info('Dumping model to {}'.format(filename))
    with open(filename, 'w') as f:
        f.write(json.dumps(model))
    logger.info('Model dumped')


### Load old job for continuation or start new

# In[ ]:

# quick'n'dirty numpy<->json encoding/decoding
def json_encode_hists(d):
    return {k:{k:{k:v.tolist() for k,v in v.items()} for k,v in v.items()} for k, v in d.items()}
def json_decode_hists(d):
    return {int(k):{str(k):{str(k):np.array(v) for k,v in v.items()} for k,v in v.items()} for k, v in d.items()}
def json_encode_perfs(d):
    return d
def json_decode_perfs(d):
    return {int(k):{str(k):{str(k):v for k,v in v.items()} for k,v in v.items()} for k, v in d.items()}

if env['OLD_JOB']:
    logger.info("Loading data for OLD_JOB: {}".format(env['OLD_JOB']))
    
    logger.debug('Loading histograms')
    with open(old_out_prefix + 'histograms.json') as f:
        hists = json_decode_hists(json.load(f))

    logger.debug('Loading perfomance metrics')
    with open(old_out_prefix + 'performances.json') as f:
        perfs = json_decode_perfs(json.load(f))
    
    assert sorted(hists.keys()) == sorted(perfs.keys())
    completed_epochs = max(hists.keys())
    
    load_model(
        cos_net,
        (old_out_prefix + 'model{:04d}.json').format(completed_epochs),
    )
        
else:
    logger.info('Starting job from scratch (No OLD_JOB given)')
    hists = {0: get_hists()}
    perfs = {0: perf_eval()}
    # Model already randomly initialised
    completed_epochs = 0


### Train

# In[ ]:

logger.info('Pre-training evaluation (on random model weights/loaded model as per above)')
logger.debug("Performance evaluation before training: {}".format(json.dumps(perfs[completed_epochs])))
logger.info("  training error:\t\t{:.20f}".format(perfs[completed_epochs]['training']['error']))
logger.info("  training accuracy:\t\t{:.2f} %".format(perfs[completed_epochs]['training']['accuracy'] * 100))
logger.info("  validation error:\t\t{:.20f}".format(perfs[completed_epochs]['validation']['error']))
logger.info("  validation accuracy:\t\t{:.2f} %".format(perfs[completed_epochs]['validation']['accuracy'] * 100))

logger.info("Training for {} epochs on top of {} old epochs".format(env['EPOCHS'], completed_epochs))
for epoch, completed_epochs in enumerate(range(
        completed_epochs+1, 
        completed_epochs+env['EPOCHS']+1
    )):
    train_mbatches = 0
    start_epoch = datetime.datetime.now()
    with Timer('Shuffle, epoch {}'.format(epoch)):
        pairs_train = shuffle(pairs_train)
    for mbatch in iterate_minibatches(pairs_train, env['BATCH_SIZE'], env['MAX_PAIRS']):
        start_mbatch = datetime.datetime.now()
        with Timer('Train epoch {}'.format(epoch)):
            train_fn(*mbatch)
        train_mbatches += 1
        n_pairs_mbatch = mbatch[0].shape[0]
        speed = n_pairs_mbatch/(datetime.datetime.now()-start_mbatch).total_seconds()
        logger.debug('Minibatch completed. speed=%d [pairs/sec]' % speed)

    with Timer('Validation epoch {}'.format(epoch)):
        perfs[completed_epochs] = perf_eval()
    logger.debug("Performance evaluation after epoch {}({} total): {}".format(
            epoch, completed_epochs, json.dumps(perfs[completed_epochs])))
    logger.info("  training error:\t\t{:.20f}".format(perfs[completed_epochs]['training']['error']))
    logger.info("  training accuracy:\t\t{:.2f} %".format(perfs[completed_epochs]['training']['accuracy'] * 100))
    logger.info("  validation error:\t\t{:.20f}".format(perfs[completed_epochs]['validation']['error']))
    logger.info("  validation accuracy:\t\t{:.2f} %".format(perfs[completed_epochs]['validation']['accuracy'] * 100))

    hists[completed_epochs] = get_hists()
    dump_model(cos_net, out_prefix + 'model' + str(completed_epochs).zfill(4)+ '.json')

    end_epoch = datetime.datetime.now()
    dur_epoch = end_epoch-start_epoch
    logger.info("Completed epoch %s of %d, time=%.3f[sec]"                 % (epoch + 1, env['EPOCHS'], dur_epoch.total_seconds()))
    logger.info('Timer(Epoch)\t\t%s' % dur_epoch)
    
    if datetime.datetime.now() + dur_epoch > end_script_before:
        logger.warning("Skipping any remaining epochs as last epoch took longer than remaining time")
        break

logger.info('Training complete')


### Performance metrics

# In[ ]:

logger.info('Dumping histograms')
with open(out_prefix + 'histograms.json', 'w') as f:
    json.dump(json_encode_hists(hists), f)
    
logger.info('Dumping perfomance metrics')
with open(out_prefix + 'performances.json', 'w') as f:
    json.dump(json_encode_perfs(perfs), f)

logger.info('Plotting histograms')
plot_hists(hists)

logger.info('Plotting performance')
plot_perfs(perfs)


### Analyse errors in correlation detection

# In[ ]:

# count errors
error_dict = dict()

land = np.logical_and
lnot = np.logical_not

for mbatch in iterate_minibatches(pairs_val, env['BATCH_SIZE'], env['MAX_PAIRS'], include_incidents=True):
    alerts1, alerts2, masks1, masks2, corelations, incidents1, incidents2 = mbatch
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
        incidents1, incidents2
    ):
        error_dict[i] = error_dict.get(i, np.zeros(4)) + np.array([tp, tn, fp, fn])
        error_dict[j] = error_dict.get(j, np.zeros(4)) + np.array([tp, tn, fp, fn])
(labels, errors) = zip(*sorted(list(error_dict.items())))
# Proper label for benign
assert labels[0] == -1, "Expecting first label to be benign, encoded by -1"
labels = tuple(['benign']) + labels[1:]

cols = ['TP', 'TN', 'FP', 'FN']
errors = pd.DataFrame(np.array(errors), columns=cols, index=labels, dtype=int)
errors.loc['total'] = errors.sum()

# Normalised errors
cols_norm = [c + ' (Norm.)' for c in cols]
errors[cols_norm] = (errors[cols].T / errors[cols].sum(axis=1)).T

# error rates
cols_rate = ['TPR', 'TNR', 'FPR', 'FNR']
errors['TPR'] = errors['TP'].astype(float) / (errors['TP'] + errors['FN'])
errors['TNR'] = errors['TN'].astype(float) / (errors['TN'] + errors['FP'])
errors['FPR'] = errors['FP'].astype(float) / (errors['TN'] + errors['FP'])
errors['FNR'] = errors['FN'].astype(float) / (errors['TP'] + errors['FN'])


# In[ ]:

logger.debug('Errors table for latex: ' + errors[cols].to_latex())
logger.info('Errors table:\n'+ errors[cols].to_string())
errors[cols]


# In[ ]:

logger.debug('Normalised errors table for latex: ' + errors[cols_norm].to_latex())
logger.info('Normalised errors table:\n'+ errors[cols_norm].to_string())
errors[cols_norm]


# In[ ]:

logger.debug('Normalised errors table for latex: ' + errors[cols_norm].to_latex())
logger.info('Normalised errors table:\n'+ errors[cols_norm].to_string())
errors[cols_norm]


# In[ ]:

# Constants for plotting
index_x = np.arange(len(labels))
bar_width = 0.2
colors = ['g', 'b', 'r', 'y']


# In[ ]:

fig, ax = plt.subplots()
for x, (metric, color) in enumerate(zip(cols_norm, colors)):
    y = errors[metric].iloc[:-1] # skip total
    rect = plt.bar(
        index_x + bar_width * x,
        y,
        bar_width,
        alpha=0.8,
        color=color,
        error_kw={'ecolor': '0.3'},
        label=metric,
    )
plt.title('Detection outcomes pr. incident (Normalised)')
plt.xlabel('Incident')
plt.ylabel('Rate')
plt.ylim(0, 1)
plt.xticks(index_x + bar_width, labels)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(out_prefix+'detection_norm.pdf', bbox_inches='tight')


# In[ ]:

fig, ax = plt.subplots()
for x, (metric, color) in enumerate(zip(cols, colors)):
    y = errors[metric].iloc[:-1] # skip total
    rect = plt.bar(
        index_x + bar_width * x,
        y,
        bar_width,
        alpha=0.8,
        color=color,
        error_kw={'ecolor': '0.3'},
        label=metric,
    )
plt.title('Detection outcomes pr. incident')
plt.xlabel('Incident')
plt.ylabel('Count (Pairs)')
plt.xticks(index_x + bar_width, labels)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(out_prefix+'detection_notnorm.pdf', bbox_inches='tight')


# In[ ]:

logger.info('Complete error table:\n'+errors.to_string())
logger.debug('Complete error table, latex:\n'+errors.to_latex())
errors.to_csv(out_prefix + 'errors.csv')


## Clustering

# In[ ]:

def _get_alert_batch(data, cut, max_samples=0):
    """
    Encode and structures single alerts similarly to pairs

    Similar to _get_batch, but ommitting second alerts and correlation.
    Currently, wont do batching.
    """
    data = shuffle(data[data['cut'] == cut])
    if max_samples:
        data = data.head(max_samples)
    alerts = np.array(data['encoded_alert'].values.tolist())
    masks = np.array(data['mask'].values.tolist())
    incidents = np.array(data['incident'].values.tolist())
    return alerts, masks, incidents

def precompute_distance_matrix(X):
    """
    precomputing takes 20 sec/500 samples on js3, OMP_NUM_THREADS=16
    precomputing takes 9 min/2632 samples on js3, OMP_NUM_THREADS=16
    """
    with Timer('Precomputing distances for {} samples'.format(len(X))):
        precomp_dist = np.zeros(shape=(len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                precomp_dist[i, j] = sp.spatial.distance.cosine(X[i], X[j])
    return precomp_dist


# In[ ]:

logger.info('Getting alerts for clustering')
clust_alerts = {
    'train': _get_alert_batch(data, 0),
    'validation': _get_alert_batch(data, 1),
    'test': _get_alert_batch(data, 2),
}

logger.info("Precomputing clustering alert distances")

X = dict()
X_dist_precomp = dict()
y = dict()
for (cut, v) in clust_alerts.items():
    alerts_matrix, masks_matrix, incidents_vector = v
    X[cut] = alert_to_vector(alerts_matrix, masks_matrix)
    X_dist_precomp[cut] = precompute_distance_matrix(X[cut])
    y[cut] = incidents_vector


# In[ ]:

for cut in y.keys():
    logger.info("Breakdown of {} labels:\n".format(cut) +  break_down_data(y[cut]))


### Cluster train data

# In[ ]:

from sklearn.cluster import DBSCAN
from sklearn import metrics

def cluster(eps, min_samples, X_dist_precomp):
    return DBSCAN(
        eps=eps, min_samples=min_samples, metric='precomputed'
    ).fit(X_dist_precomp)

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

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        for i, x_core in enumerate(X['train'][dbscan_model.core_sample_indices_]): 
            if  metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


# In[ ]:

logger.info("Iterating clustering algorithm parameters")
epss = np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])
min_sampless = np.array([1, 3, 10, 30])

def ParamSpaceMatrix(dtype=None):
    return np.zeros(shape=(len(epss), len(min_sampless)), dtype=dtype)

cl_model = ParamSpaceMatrix(dtype=object)
mapper = ParamSpaceMatrix(dtype=object)

cuts = ['train', 'validation']

def ParamSpaceMatrices(dtype=None):
    return {cut : ParamSpaceMatrix(dtype=dtype) for cut in cuts}


# In[ ]:

# Cluster and build mapper
for i, eps in enumerate(epss):
    for j, min_samples in enumerate(min_sampless):
        logger.info("Clustering, eps={}, min_samples={}".format(eps, min_samples))
        # Cluster
        cl_model[i,j] = cluster(eps, min_samples, X_dist_precomp['train'])
        # get cluster assignments
        y_pred = cl_model[i,j].labels_ 
        # Build classifier - get mapper used for classification
        mapper[i,j] = build_cluster_to_incident_mapper(y['train'], y_pred)


# In[ ]:

# predict 
y_pred = ParamSpaceMatrices(dtype=object)
y_pred_inc = ParamSpaceMatrices(dtype=object)

for cut in cuts:
    for i, eps in enumerate(epss):
        for j, min_samples in enumerate(min_sampless):
            if cut == 'train':
                # pred is abused to hold clustering results
                y_pred[cut][i,j] = cl_model[i,j].labels_ # cluster assignment
                y_pred_inc[cut][i,j] = y[cut] # true incident label
            elif cut == 'validation':
                logger.info('Predicting for (eps, min_samples)=({:1.0e},{:>2d})'.format(eps, min_samples))
                y_pred[cut][i,j] = dbscan_predict(cl_model[i][j], X[cut])
                y_pred_inc[cut][i,j] = np.array([mapper[i,j][el] for el in y_pred[cut][i,j]]) # predict incident
            else:
                raise NotImplementedError('Unexpected value for cut:{}'.format(cut))
            


# In[ ]:

def false_alert_rate_outliers_score(y, y_pred):
    idx_outliers = y_pred == -1
    return (y[idx_outliers] == -1).mean()

def arf_score(y, y_pred):
    return len(y) / len(set(y_pred))

def narf_score(y, y_pred):
    return (len(y) / len(set(y_pred)) - 1) / (len(y) - 1)

def imr_score(y, y_pred):
    df_clustering = pd.DataFrame({
        'cluster': y_pred,
        'inc_true': y,
    })
    cluster_sizes = pd.DataFrame({'cluster_size': df_clustering[df_clustering.cluster != -1].groupby('cluster').size()})
    df_clustering = pd.merge(df_clustering, cluster_sizes.reset_index(), on='cluster', how='outer')
    # asuming one alert is picked at random from each cluster;
    # probability that given alert is picked to represent the cluster it belongs to
    df_clustering['alert_pick_prob'] = 1/df_clustering.cluster_size 
    # probability distribution of what incident a cluster will be asumed to represent
    df_prob = pd.DataFrame(df_clustering.groupby(['cluster', 'inc_true']).sum().alert_pick_prob.rename('inc_hit'))
    # probability that a given incident will not come out of a cluster
    df_prob['inc_miss'] = 1 - df_prob.inc_hit.fillna(0)
    assert (df_prob[df_prob.inc_miss < 0].inc_miss.abs() < 1e-12).all(), "Error larger than 1e-12, still just imprecission?"
    df_prob = df_prob.abs()
    # ... of any cluster
    inc_miss_prob = df_prob.reset_index().groupby('inc_true').inc_miss.prod().rename('inc_miss_prob')
    inc_miss_prob = inc_miss_prob[inc_miss_prob.index != -1] # Don't care about missing the noise pseudo-incident
    return inc_miss_prob.sum() / df_clustering[df_clustering.inc_true != -1].inc_true.unique().shape[0]


# In[ ]:

# calculating metrics

# clustering
n_clusters = ParamSpaceMatrices(dtype=int)
homogenity = ParamSpaceMatrices()
noise = ParamSpaceMatrices(dtype=int)
noise_false_rate = ParamSpaceMatrices()
# classification, general
accuracy = ParamSpaceMatrices()
precision = ParamSpaceMatrices()
recall = ParamSpaceMatrices()
f1 = ParamSpaceMatrices()
# correlating and filtering metrics
arf = ParamSpaceMatrices()
narf = ParamSpaceMatrices()
imr = ParamSpaceMatrices()
faro = ParamSpaceMatrices()

for cut in cuts:
    for i, eps in enumerate(epss):
        for j, min_samples in enumerate(min_sampless):
            # clustering metrics
            # Number of clusters in labels, ignoring noise if present.
            n_clusters[cut][i,j] = len(set(y_pred[cut][i,j])) - (1 if -1 in y_pred[cut][i,j] else 0)
            noise[cut][i,j] = sum(y_pred[cut][i,j] == -1)
            homogenity[cut][i,j] = metrics.homogeneity_score(y[cut], y_pred[cut][i,j])
            # classification metrics
            accuracy[cut][i,j] = metrics.accuracy_score(y[cut], y_pred_inc[cut][i,j])
            precision[cut][i,j] = metrics.precision_score(y[cut], y_pred_inc[cut][i,j], average='weighted')
            recall[cut][i,j] = metrics.recall_score(y[cut], y_pred_inc[cut][i,j], average='weighted')
            f1[cut][i,j] = metrics.f1_score(y[cut], y_pred_inc[cut][i,j], average='weighted')
            # correlating and filtering
            arf[cut][i,j] = arf_score(y[cut], y_pred[cut][i,j])
            narf[cut][i,j] = narf_score(y[cut], y_pred[cut][i,j])
            imr[cut][i,j] = imr_score(y[cut], y_pred[cut][i,j])
            faro[cut][i,j] = false_alert_rate_outliers_score(y[cut], y_pred[cut][i,j])

            logger.info(
                "Performance on {} cut with (eps, min_samples)=({:1.0e},{:>2d}): n_clusters={:>3d}, homogenity={:1.3f}, f1={:1.3f}, noise={:>3d}".format(
                    cut, eps, min_samples, n_clusters[cut][i,j], homogenity[cut][i,j], f1[cut][i,j], noise[cut][i,j],
                )
            )
        


# In[ ]:

def param_plot_prepare(
    title,
):
    fig, ax = plt.subplots(figsize=(10,6))
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

for label, values, fmt in [
    ('Cluster count', n_clusters, '{}'),
    ('Cluster homogenity', homogenity, '{:.2f}'),
    ('Noise (Clustering outliers)', noise, '{}'),
]:
    for cut in cuts:
        param_plot_prepare('{} by DBSCAN parameters ({} alerts)'.format(label, cut))
        param_plot_scatter(values[cut], epss, min_sampless)
        param_plot_annotate(values[cut], epss, min_sampless, fmt=fmt)
        
        param_plot_save(
            '{}{}_{}.pdf'.format(
                out_prefix,
                label.replace('(','').replace(')','').replace(" ", "_").lower(),
                cut
            )
        )


# In[ ]:

for label, values, fmt in [
    ('Incident prediction accuracy', accuracy, '{:.2f}'),
    ('Incident prediction precision', precision,  '{:.2f}'),
    ('Incident prediction recall', recall, '{:.2f}'),
    ('Incident prediction F1 score', f1, '{:.2f}'),
]:
    for cut in ['validation']: # training evaluated on true labels for clustering is of little use
        param_plot_prepare('{} by DBSCAN parameters ({} alerts)'.format(label, cut))
        param_plot_scatter(values[cut], epss, min_sampless)
        param_plot_annotate(values[cut], epss, min_sampless, fmt=fmt)
        
        param_plot_save(
            '{}{}_{}.pdf'.format(
                out_prefix,
                label.replace('(','').replace(')','').replace(" ", "_").lower(),
                cut
            )
        )


# In[ ]:

for label, values, fmt in [
    ('Alert Reduction Factor', arf, '{:.2f}'),
    ('Normalised Alert Reduction Factor', narf, '{:.2e}'),
    ('Incident Miss Rate', imr, '{:.2e}'),
    ('False alerts rate among outliers', faro, '{:.2f}'),
]:
    for cut in cuts:
        param_plot_prepare('{} by DBSCAN parameters ({} alerts)'.format(label, cut))
        param_plot_scatter(values[cut], epss, min_sampless)
        param_plot_annotate(values[cut], epss, min_sampless, fmt=fmt)
        
        param_plot_save(
            '{}{}_{}.pdf'.format(
                out_prefix,
                label.replace('(','').replace(')','').replace(" ", "_").lower(),
                cut
            )
        )


# In[ ]:

def cm_inc_clust(y, y_pred):
    cm_inc_clust = pd.DataFrame(
        metrics.confusion_matrix(y, y_pred),
        index=sorted(set.union(set(y), set(y_pred))),
        columns=sorted(set.union(set(y), set(y_pred))),
    )
    # drop dummy row for non-existing incident IDs
    assert (cm_inc_clust.drop(list(set(y)), axis=0) == 0).as_matrix().all(), "Non-empty row for invalid incident id"
    cm_inc_clust = cm_inc_clust.loc[sorted(list(set(y)))]

    # drop dummy collumns for non-existing cluster IDs
    assert (cm_inc_clust.drop(list(set(y_pred)), axis=1) == 0).as_matrix().all(), "Non-empty collumn for invalid cluster id"
    cm_inc_clust = cm_inc_clust[sorted(list(set(y_pred)))]

    cm_inc_clust.rename(index={-1: 'benign'}, inplace=True)
    cm_inc_clust.rename(columns={-1: 'noise'}, inplace=True)
    return cm_inc_clust

def cm_inc_inc(y, y_pred_inc):
    cm_inc_inc = pd.DataFrame(
        metrics.confusion_matrix(y, y_pred_inc),
        index=sorted(set.union(set(y), set(y_pred_inc))),
        columns=sorted(set.union(set(y), set(y_pred_inc))),
    )

    cm_inc_inc.rename(index={-1: 'benign'}, inplace=True)
    cm_inc_inc.rename(columns={-1: 'benign'}, inplace=True)
    return cm_inc_inc



### Clustering - test data

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

alerts_matrix, masks_matrix, incidents_vector = clust_alerts_test
X = alert_to_vector(alerts_matrix, masks_matrix)
y = incidents_vector

y_pred = dbscan_predict(cl_model[i][j], X)
y_pred_inc = np.array([mapper[i][j][el] for el in y_pred])


# In[ ]:

sorted(set.union(set(y), set(y_pred)))


# In[ ]:

logger.info("Incident (i) to cluster (j) \"confusion matrix\":\n" + cm_inc_clust.to_string())
logger.debug("Incident (i) to cluster (j) \"confusion matrix\" in latex:\n" + cm_inc_clust.to_latex())
cm_inc_clust


# In[ ]:

logger.info("Incident (i) to Incident (j) confusion matrix:\n" + cm_inc_inc.to_string())
logger.debug("Incident (i) to Incident (j) confusion matrix in latex:\n" + cm_inc_inc.to_latex())
cm_inc_inc


# In[ ]:

cm = metrics.confusion_matrix(y, y_pred_inc)
logger.info("Classification accuracy: {:.2f}%".format(
    cm_inc_inc.as_matrix().diagonal().sum() / cm_inc_inc.as_matrix().sum() * 100
))


# In[ ]:

report = pd.DataFrame(
    dict(zip(
        ['precision', 'recall', 'f1-score', 'support'],
        metrics.precision_recall_fscore_support(y, y_pred_inc),
    )),
    index=sorted(list(set(y))),
)[['precision', 'recall', 'f1-score', 'support']]
mean = report.mean(axis=0).drop('support')
zum = report.sum(axis=0).drop(['precision', 'recall', 'f1-score'])

report.loc['mean'] = mean
report.loc['sum'] = zum
report['support'] = report['support'].fillna(-1).astype(int)

logger.info("Classification report:\n"+report.to_string())
logger.debug("Classification report in latex:\n"+report.to_latex())
report


# In[ ]:

logger.info('Testing completed, exiting')
sys.exit(0)


### Analysing results

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



