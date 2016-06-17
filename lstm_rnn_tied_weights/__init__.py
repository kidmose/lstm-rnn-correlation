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
from sklearn.utils.extmath import cartesian
import logging
import random
import netaddr
import re

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if len(logger.handlers) == 0:
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter())
        logger.addHandler(ch)

        # # File
        # fh = logging.FileHandler(__name__+'.log')
        # fh.setLevel(logging.DEBUG)
        # fh.setFormatter(logging.Formatter(
        #     fmt='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
        # ))
        # logger.addHandler(fh)

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
def load(
        file_names,
):
    """
    Loads incidents from the files listed in file_names, enumerates and returns a dict
    """
    start_time = time.time()
    logger.info("Loading {} files:".format(len(file_names)))

    incidents = list()
    for i, fn in enumerate(file_names):
        i += 1
        alerts = list()
        logger.info(' - {}/{} {}'.format(i, len(file_names), fn))
        with open(fn, 'r') as f:
            for l in f.readlines():
                alerts.append(l)
        incidents.append((i, alerts))

    logger.info("Completed loading {} alerts in {}s".format(
        sum(map(lambda incident: len(incident[1]), incidents)),
        time.time()-start_time),
    )
    return incidents

def modify(
        incidents,
        modifier_fns,
):
    """
    Modifies alerts in incidents by executing modifier function on them,
    """
    for fn in modifier_fns:
        logger.info('Applying modifier function: {}'.format(fn))
        incidents = fn(incidents)
    return incidents

def pool(
        incidents,
):
    """
    Pools incidents into one list of alerts.

    incidents: dict with incident ids mapping to list of alert strings
    returns: list of (<incident id>, <alert text string>) tuples
    """
    def gen():
        for incident_alerts in incidents:
            incident, alerts = incident_alerts
            for alert in alerts:
                yield (incident, alert)
    return list(gen())

def encode(alerts):
    """
    Encodes alerts as one hot encoding for ascii chars and calculates masks.

    alerts: list of (<incident id>, <alert text string>) tuples

    returns: (alert_matrix, mask_matrix, incidents)
    alert_matrix: numpy matrix of size 'alert count' by 'max length among alerts' with ascii(int8) encoded strings.
    mask_matrix: Dimensions like alert_matrix. Encoding length of alerts with one of {0,1} pr. character.
    incident: list an entry for each row in alert_matrix, identifying source file for the row.
    """
    incidents, alerts = zip(*alerts)
    lens = list(map(len, alerts))
    alert_matrix = np.zeros((len(alerts),max(lens)), dtype='int8')
    mask_matrix = np.zeros_like(alert_matrix)
    for i, alert in enumerate(alerts):
        mask_matrix[i, :lens[i]] = 1
        for j, c in enumerate(alert):
            alert_matrix[i,j] = ord(c)

    return (alert_matrix, mask_matrix, incidents)

def split(
        samples,
        split,
):
    """
    Split data into training, validation and test sets.

    Applies a deterministic shuffle on input before splitting.
    """
    assert len(split) == 3
    n = len(samples)

    weights = np.array(split)
    weights = (weights/sum(weights)*n).astype(int)
    idxs = list(np.cumsum(weights))
    idxs = [0] + idxs

    random.seed(1131662768)
    random.shuffle(samples)

    logger.info("Splitting {} samples into: ".format(len(samples)) + str(weights))

    return tuple((
        samples[start:end]
        for start, end
        in zip (idxs[:-1], idxs[1:])))


def cross_join_forfor(n, offset):
    """
    [BUGGED] Cross join implemented with two nested for loops.

    Scales well as it does not produce entire matrix, but see "BUG"

    BUG: Looping over two index arrays fails to shuffle uniformly;
    First n pairs will have the same alert in position one,
    next n pairs also and so on.
    """
    logger.info("Cross join with two for loops is used")
    logger.warn("Using buggy implementation: Not truly random")
    # Shuffle to sample across all alerts in a predictive fashion
    np.random.seed(1131662768)
    i_idxs = np.arange(n)
    np.random.shuffle(i_idxs)
    j_idxs = np.arange(n)
    np.random.shuffle(j_idxs)

    # Handle offsetting
    i_offset = offset // len(alerts)
    j_offset = offset % len(alerts)
    i = i_idxs[i_offset] # find first row to be used
    for j in j_idxs[j_offset:]: # find elements to be used
        yield produce_one_pair(i, j)

    # remainder is straight forward
    for i in i_idxs[i_offset+1:]:
        for j in j_idxs:
            yield (i, j)


def cross_join_python(n, offset):
    """
    Cross join implemented with python core methods.

    Efficient in time, requires list of n**2 index pairs in memory.
    """
    logger.info("Cross join with python shuffle is used")

    idxs = [(i,j) for i in range(n) for j in range(n)]
    random.seed(1131662768)
    random.shuffle(idxs)
    for ij in idxs[offset:]:
        yield ij


def cross_join_numpy(n, offset):
    """
    Cross join implemented with numpy methods.

    Efficient in time, requires list of n**2 index pairs in memory.
    """
    logger.info("Cross join with numpy shuffle is used")

    idxs = cartesian([np.arange(n), np.arange(n)])
    np.random.seed(1131662768)
    np.random.shuffle(idxs)
    for ij in idxs[offset:]:
        yield ij


def cross_join_rand_samp(n, offset):
    """
    [NotImplemented] Cross joing implemented with random sampling.
    """
    raise NotImplementedError("Please see recent commit for how this can be done")


def cross_join(
        alerts,
        offset=0,
        implementation=cross_join_numpy,
):
    """
    Creates pairs from provided alerts with the provided implementation.
    """
    alerts, masks, incidents = encode(alerts)

    n = len(alerts)
    assert n == len(masks)
    assert n == len(incidents)
    assert alerts.shape == masks.shape

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

    logger.info(
        "Making pairs with {}, using {} alerts. Result is offset with {}"
        .format(implementation, n, offset)
    )
    for cnt, (i, j) in enumerate(implementation(n, offset)):
        yield produce_one_pair(i, j)
    logger.info('Sucesfully made {} pairs of alerts (implementation empty)'.format(cnt+1))


def limit(iterable, max_samples):
    logger.debug('Limitting to max_samples={}'.format(max_samples))
    for sample, cnt in zip(iterable, range(max_samples)):
        yield sample
    cnt += 1 # from zero to 1 indexed
    if cnt != max_samples:
        logger.warn("Iterable empty before max_samples read")
    logger.debug('Limited to {} pairs (max_samples={})'.format(cnt, max_samples))


def iterate_minibatches(samples, batch_size, keep_incidents=False):
    # Sneek peak at first sample to learn alert length
    sample = next(samples)
    alert1, alert2, mask1, mask2, correlation, incident1, incident2 = sample
    inputs1 = np.empty((batch_size, len(sample[0])), dtype=alert1.dtype)
    inputs2 = np.empty_like(inputs1)
    masks1 = np.empty_like(inputs1)
    masks2 = np.empty_like(inputs1)
    targets = np.empty((batch_size), dtype=bool)
    incs1 = np.empty_like(targets, dtype=int)
    incs2 = np.empty_like(targets, dtype=int)
    i = 0 # index for the arrays

    # Remember to use first sample
    inputs1[i], inputs2[i], masks1[i], masks2[i], targets[i], incs1[i], incs2[i] = sample
    i += 1

    # Use remaining samples to build and yield batches
    batches_produced = 0
    samples_processed = 0
    for sample in samples:
        inputs1[i], inputs2[i], masks1[i], masks2[i], targets[i], incs1[i], incs2[i] = sample
        i += 1
        if i == batch_size:
            batches_produced += 1
            samples_processed += i
            i = 0
            logger.debug('Yielding batch, len={}'.format(len(inputs1)))
            if keep_incidents:
                logger.debug('Yielding batch with incident information')
                yield inputs1, inputs2, masks1, masks2, targets, incs1, incs2
            else:
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

# Data modification methods used in modify
def replace_re_in_alerts(alerts, pattern, new):
    fn = lambda alert : re.sub(pattern, new, alert)
    return map(fn, alerts)

PATTERN_IP = '(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
PATTERN_TS = '[0-9]{2}/[0-9]{2}-[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}'
PATTERN_PORT = '(<IP>|'+PATTERN_IP+'):[0-9]+'

def mask_ips(incidents):
    logger.info('Masking out IP addresses')
    return [
        (incidentid, replace_re_in_alerts(alerts, PATTERN_IP, '<IP>'))
        for incidentid, alerts in incidents
    ]

def mask_tss(incidents):
    logger.info('Masking out timestamps')
    return [
        (incidentid, replace_re_in_alerts(alerts, PATTERN_TS, '<TIMESTAMP>'))
        for incidentid, alerts in incidents
    ]

def mask_ports(incidents):
    logger.info('Masking out ports')
    return [
        (incidentid, replace_re_in_alerts(alerts, PATTERN_PORT, '<IP>'))
        for incidentid, alerts in incidents
    ]

def uniquify_victim(incidents, oldips):
    """
    Replaces specified IPs in incidents with random ones
    """
    logger.info("Uniquifying victims by replacing with random IPs (Same one within incident)")
    incidentids, alertlists = zip(*incidents)
    assert len(set(incidentids)) == len(incidents), "incidents ids must be unique"
    assert isinstance(oldips, list), "IP to replace must be list"
    assert len(set(incidentids)) == len(oldips), "requires one IP pr incident"

    used = set(oldips)
    def get_random_ip(used, seed=None):
        np.random.seed(seed)
        ip = netaddr.IPAddress('.'.join(
            [str(octet) for octet in np.random.randint(256, size=(4))]
        ))
        if str(ip) not in used and ip.is_unicast() and not ip.is_loopback():
            return str(ip)
        else:
            return get_random_ip(used)
    newips = dict() # {incident id : new IP}
    for i in incidentids:
        newips[i] = get_random_ip(used, seed=i)
        used.add(newips[i])
    logger.info("New IPs: " + str(newips))

    oldips_d = {iid: oip for (iid, oip) in zip(incidentids, oldips)}
    return [
        (incidentid, replace_re_in_alerts(alerts, oldips_d[incidentid], newips[incidentid]))
        for incidentid, alerts in incidents
    ]

prio_from_alert = lambda alert: int(re.match('.*\[Priority: ([0-9]+)\].*', alert).group(1))
prio_from_alerts = lambda alerts: map(prio_from_alert, alerts)

def extract_prio(incidents):
    logger.info('Extracting priority from alerts in incidents')
    return [
        (incidentid, map(prio_from_alert, alerts))
        for incidentid, alerts in incidents
    ]

def get_discard_by_prio(key = lambda prio : prio < float('Inf')):
    def discard_by_prio(incidents):
        logger.info('Discarding alerts by priority')
        return [
                (incidentid, filter(
                    lambda alert: key(prio_from_alert(alert)), alerts
                ))
                for incidentid, alerts in incidents
        ]
    return discard_by_prio

def break_down_data(
    items,
    extractors = [('label', lambda item: item),]
):
    extractorsd = {label: ext for (label, ext) in extractors}
    # results as 2 layer dict {extractor : {label value : count}}
    results = {e: dict() for e in extractorsd.keys()}
    # do the counting
    for item in items:
        for ext, cnt_dict in results.items():
            value = extractorsd[ext](item)
            cnt_dict[value] = cnt_dict.get(value, 0) + 1
    # Format result
    retval = str()
    for ext_label in [ext_label for (ext_label, ext) in extractors] :
        label, cnt = zip(*results[ext_label].items())
        cnt = np.array(cnt)
        cnt_norm = cnt/float(sum(cnt))*100
        retval += "Breakdown of {}:\n".format(ext_label)
        retval += 'Label: '+('{: >10}'*len(label)).format(*label) + '\n'
        retval += 'Count: '+('{: >10}'*len(label)).format(*cnt) + '\n'
        retval += 'Norm.: '+('{: >9.2f}%'*len(label)).format(*cnt_norm) + '\n'
    return retval
