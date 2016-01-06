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
"""
Module for LSTM RNN with tied weights.
"""
from __future__ import division

from lasagne import layers
import theano.tensor as T

import time
import math
import numpy as np
import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if len(logger.handlers) == 0:
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter())
        logger.addHandler(ch)

        # File
        fh = logging.FileHandler(__name__+'.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            fmt='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
        ))
        logger.addHandler(fh)

    return logger

logger = get_logger('lstm_rnn_tied_weights')

class CosineSimilarityLayer(layers.MergeLayer):
    """
    Cosine simlarity for neural networks in lasagne.

    $\cos(\theta_{A,B}) = {A \cdot B \over \|A\| \|B\|} = \frac{ \sum\limits_{i=1}^{n}{A_i \times B_i} }{ \sqrt{\sum\limits_{i=1}^{n}{(A_i)^2}} \times \sqrt{\sum\limits_{i=1}^{n}{(B_i)^2}} }$
    """
    def __init__(self, incoming1, incoming2, **kwargs):
        """Instantiates the layer with incoming1 and incoming2 as the inputs."""
        incomings = [incoming1, incoming2]

        for incoming in incomings:
            if isinstance(incoming, tuple):
                if len(incoming) != 2:
                    raise NotImplementedError("Requires shape to be exactly (BATCH_SIZE, N).")
            elif len(incoming.output_shape) != 2:
                raise NotImplementedError("Requires shape to be exactly (BATCH_SIZE, N).")

        super(CosineSimilarityLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        """Return output shape: (batch_size, 1)."""
        if len(input_shapes) != 2:
            raise ValueError("Requires exactly 2 input_shapes")

        for input_shape in input_shapes:
            if len(input_shape) != 2:
                raise NotImplementedError("Requires shape to be exactly (BATCH_SIZE, N).")

        return (input_shape[0],)

    def get_output_for(self, inputs, **kwargs):
        """Calculates the cosine similarity."""
        nominator = (inputs[0] * inputs[1]).sum(axis=1)
        denominator = T.sqrt((inputs[0]**2).sum(axis=1)) * T.sqrt((inputs[1]**2).sum(axis=1))
        return nominator/denominator


def clone(src_net, dst_net, mask_input):
    """
    Clones a lasagne neural network, keeping weights tied.

    For all layers of src_net in turn, starting at the first:
     1. creates a copy of the layer,
     2. reuses the original objects for weights and
     3. appends the new layer to dst_net.

    InputLayers are ignored.
    Recurrent layers (LSTMLayer) are passed mask_input.
    """
    logger.info("Net to be cloned:")
    for l in layers.get_all_layers(src_net):
        logger.info(" - {} ({}):".format(l.name, l))

    logger.info("Starting to clone..")
    for l in layers.get_all_layers(src_net):
        logger.info("src_net[...]: {} ({}):".format(l.name, l))
        if type(l) == layers.InputLayer:
            logger.info(' - skipping')
            continue
        if type(l) == layers.DenseLayer:
            dst_net = layers.DenseLayer(
                dst_net,
                num_units=l.num_units,
                W=l.W,
                b=l.b,
                nonlinearity=l.nonlinearity,
                name=l.name+'2',
            )
        elif type(l) == layers.EmbeddingLayer:
            dst_net = layers.EmbeddingLayer(
                dst_net,
                l.input_size,
                l.output_size,
                W=l.W,
                name=l.name+'2',
            )
        elif type(l) == layers.LSTMLayer:
            dst_net = layers.LSTMLayer(
                dst_net,
                l.num_units,
                ingate=layers.Gate(
                    W_in=l.W_in_to_ingate,
                    W_hid=l.W_hid_to_ingate,
                    W_cell=l.W_cell_to_ingate,
                    b=l.b_ingate,
                    nonlinearity=l.nonlinearity_ingate
                ),
                forgetgate=layers.Gate(
                    W_in=l.W_in_to_forgetgate,
                    W_hid=l.W_hid_to_forgetgate,
                    W_cell=l.W_cell_to_forgetgate,
                    b=l.b_forgetgate,
                    nonlinearity=l.nonlinearity_forgetgate
                ),
                cell=layers.Gate(
                    W_in=l.W_in_to_cell,
                    W_hid=l.W_hid_to_cell,
                    W_cell=None,
                    b=l.b_cell,
                    nonlinearity=l.nonlinearity_cell
                ),
                outgate=layers.Gate(
                    W_in=l.W_in_to_outgate,
                    W_hid=l.W_hid_to_outgate,
                    W_cell=l.W_cell_to_outgate,
                    b=l.b_outgate,
                    nonlinearity=l.nonlinearity_outgate
                ),
                nonlinearity=l.nonlinearity,
                cell_init=l.cell_init,
                hid_init=l.hid_init,
                backwards=l.backwards,
                learn_init=l.learn_init,
                peepholes=l.peepholes,
                gradient_steps=l.gradient_steps,
                grad_clipping=l.grad_clipping,
                unroll_scan=l.unroll_scan,
                precompute_input=l.precompute_input,
                # mask_input=l.mask_input, # AttributeError: 'LSTMLayer' object has no attribute 'mask_input'
                name=l.name+'2',
                mask_input=mask_input,
            )
        elif type(l) == layers.SliceLayer:
            dst_net = layers.SliceLayer(
                dst_net,
                indices=l.slice,
                axis=l.axis,
                name=l.name+'2',
            )
        else:
            raise ValueError("Unhandled layer: {}".format(l))
        new_layer = layers.get_all_layers(dst_net)[-1]
        logger.info('dst_net[...]: {} ({})'.format(new_layer, new_layer.name))

    logger.info("Result of cloning:")
    for l in layers.get_all_layers(dst_net):
        logger.info(" - {} ({}):".format(l.name, l))

    return dst_net

# Data preparation / loading functions
def load_data(file_names):
    """
    Loads alerts from the files listed in file_names and returns them along with incident id.

    files_names: full or relative path to files that needs to be loaded.
    """
    start_time = time.time()
    logger.info("Loading {} files:".format(len(file_names)))

    incidents = dict()
    for i, fn in enumerate(file_names):
        i += 1
        incidents[i] = list()
        logger.info(' - {}/{} {}'.format(i, len(file_names), fn))
        with open(fn, 'r') as f:
            for l in f.readlines():
                incidents[i].append(l)

    logger.info("Completed loading {} alerts in {}s".format(
        sum(map(len, incidents.values())),
        time.time()-start_time),
    )
    return incidents


def encode(alerts, incidents):
    """
    Encodes alerts as one hot encoding for ascii chars and calculates masks.

    alerts: list of strings, each containing an alert.
    incidents: list of incidents IDs for each alert.

    returns: (alert_matrix, mask_matrix, incidents)
    alert_matrix: numpy matrix of size 'alert count' by 'max length among alerts' with ascii(int8) encoded strings.
    mask_matrix: Dimensions like alert_matrix. Encoding length of alerts with one of {0,1} pr. character.
    incident: list an entry for each row in alert_matrix, identifying source file for the row.
    """
    assert len(alerts) == len(incidents), "Inputs must have same length."
    incidents = np.array(incidents)

    lens = list(map(len, alerts))
    alert_matrix = np.zeros((len(alerts),max(lens)), dtype='int8')
    mask_matrix = np.zeros_like(alert_matrix)
    for i, alert in enumerate(alerts):
        mask_matrix[i, :lens[i]] = 1
        for j, c in enumerate(alert):
            alert_matrix[i,j] = ord(c)

    return (alert_matrix, mask_matrix, incidents)

def split_data(
    alerts,
    masks,
    incidents,
    split,
):
    """Split data into training, validation and test sets"""
    assert len(alerts) == len(masks)
    assert len(alerts) == len(incidents)
    assert alerts.shape == masks.shape
    assert len(split) == 3
    n = len(alerts)

    weights = np.array(split)
    weights = (weights/sum(weights)*n).astype(int)
    idxs = list(np.cumsum(weights))
    idxs = [0] + idxs

    logger.info("Splitting {} samples into: ".format(len(alerts)) + str(weights))
    for start, end in zip (idxs[:-1], idxs[1:]):
        yield alerts[start:end], masks[start:end], incidents[start:end]

def cross_join(
    cut,
    max_alerts=None,
    offset=0,
):
    """Cross join list of alerts with self and track if incident is the same."""
    alerts, masks, incidents = cut
    assert len(alerts) == len(masks)
    assert len(alerts) == len(incidents)
    assert alerts.shape == masks.shape

    # Shuffle to sample across all alerts in a predictive fashion
    np.random.seed(1131662768)
    i_idxs = np.arange(len(alerts))
    np.random.shuffle(i_idxs)
    j_idxs = np.arange(len(alerts))
    np.random.shuffle(j_idxs)

    def produce_one_pair(i, j):
        return (
            alerts[i],
            alerts[j],
            masks[i],
            masks[j],
            incidents[i] == incidents[j],
            incidents[i],
            incidents[j],
        )

    def produce_all_pairs():

        # Handle offsetting
        i_offset = offset // len(alerts)
        j_offset = offset % len(alerts)
        i = i_idxs[i_offset] # find first row to be used
        for j in j_idxs[j_offset:]: # find elementts to be used
            yield produce_one_pair(i, j)

        # remainder is straight forward
        for i in i_idxs[i_offset+1:]:
            for j in j_idxs:
                yield produce_one_pair(i, j)

    def limit(iterable):
        logger.debug('Limitting to max_alerts={}'.format(max_alerts))
        for pair, cnt in zip(iterable, range(max_alerts)):
            yield pair
        logger.debug('Succescully limited to {} pairs (max_alerts={})'.format(cnt, max_alerts))

    pairs = produce_all_pairs()
    if max_alerts is not None:
        pairs = limit(pairs)

    for cnt, pair in enumerate(pairs):
        yield pair
    logger.debug('Crossjoin yielded {} pairs of alerts'.format(cnt))


def iterate_minibatches(samples, batch_size):
    # Sneek peak at first sample to learn alert length
    sample = next(samples)
    alert1, alert2, mask1, mask2, correlation, incident1, incident2 = sample
    inputs1 = np.empty((batch_size, len(sample[0])), dtype=alert1.dtype)
    inputs2 = np.empty_like(inputs1)
    masks1 = np.empty_like(inputs1)
    masks2 = np.empty_like(inputs1)
    targets = np.empty((batch_size), dtype=bool)
    i = 0 # index for the arrays

    # Remember to use first samples
    inputs1[i], inputs2[i], masks1[i], masks2[i], targets[i], _, _ = sample
    i += 1

    # Use remaining samples to build and yield batches
    batches_produced = 0
    samples_processed = 0
    for sample in samples:
        inputs1[i], inputs2[i], masks1[i], masks2[i], targets[i], _, _ = sample
        i += 1
        if i == batch_size:
            batches_produced += 1
            samples_processed += i
            i = 0
            logger.debug('Yielding batch, len={}'.format(len(inputs1)))
            yield inputs1, inputs2, masks1, masks2, targets
    samples_processed += i
    logger.debug('samples_processed={}, batches_produced={}'.format(
        samples_processed, batches_produced
    ))
    assert batches_produced > 0, "{} samples is not enough to produce a batch({} samples)".format(
        samples_processed, batch_size)
    batches_expected = samples_processed // batch_size
    assert batches_expected == batches_produced, "Expected {} but produced {} batches". format(
        batches_expected, batches_produced)
