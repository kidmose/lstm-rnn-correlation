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

from lasagne import layers
import theano.tensor as T


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
    print("Net to be cloned:")
    for l in layers.get_all_layers(src_net):
        print(" - {} ({}):".format(l.name, l))

    print("Starting to clone..")
    for l in layers.get_all_layers(src_net):
        print("src_net[...]: {} ({}):".format(l.name, l))
        if type(l) == layers.InputLayer:
            print(' - skipping')
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
        print('dst_net[...]: {} ({})'.format(new_layer, new_layer.name))

    print("Result of cloning:")
    for l in layers.get_all_layers(dst_net):
        print(" - {} ({}):".format(l.name, l))

    return dst_net
