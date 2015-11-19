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

from lasagne.layers import MergeLayer
import theano.tensor as T


class CosineSimilarityLayer(MergeLayer):
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
