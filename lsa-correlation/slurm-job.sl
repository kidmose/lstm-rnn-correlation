#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --array=0-9
#SBATCH --output="lstm-rnn-correlation/output/slurm-%A_%a.out"

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

echo "Loading modules" && \
    module load python/3.6.3 && \
    echo "Loaded modules" || \
    { echo "Failed to load modules"; exit -1; }

echo "Creating node-local git repository from $HOME" && \
    cd $LOCALSCRATCH && \
    rm -rf lstm-rnn-correlation && \
    git clone $HOME/lstm-rnn-correlation && \
    cd $LOCALSCRATCH/lstm-rnn-correlation && \
    echo "Created node-local repository ($PWD;$(git rev-parse --abbrev-ref HEAD);$(git describe))" || \
    { echo "Failed to create node-local git repository"; exit -2; }

echo "Creating symlinks to data, output and env folders" && \
    ln -s $HOME/lstm-rnn-correlation/output/ output && \
    ln -s $HOME/lstm-rnn-correlation/data/ data && \
    ln -s $HOME/lstm-rnn-correlation/lsa-correlation/env/ env && \
    echo "Created symlinks" || \
    { echo "Failed to create symlinks"; exit -3; }

echo "Activating virtual environment" && \
    source env/bin/activate && \
    echo "Activated virtual environment (python version: $(python --version 2>&1) from: $(which python))" || \
    { echo "Failed to activate virtual environment"; exit -4; }

echo "Starting workload" && \
    VAL_CUT=$SLURM_ARRAY_TASK_ID RAND_SEED=$SLURM_ARRAY_TASK_ID python lsa-correlation/lsa-correlation.py && \
    echo "Workload completed sucessfully" || \
    { echo "Workload failed"; exit -5; }
