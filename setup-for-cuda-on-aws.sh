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

# based on http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/

# Update os
sudo apt-get update && sudo apt-get upgrade -y && echo "OS up to date"

# CUDA #
sudo apt-get install -y \
     build-essential \
     cmake \
     git \
     liblapack-dev \
     libopenblas-dev \
     linux-headers-generic \
     linux-image-extra-virtual \
     linux-image-generic \
     linux-source \
     pkg-config \
     unzip \
    && echo "CUDA Dependencies installed"

# disable nouveau kernel driver
echo 'blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf && \
    echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf && \
    sudo update-initramfs -u && \
    sudo reboot

wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run && \
    mkdir installers && \
    chmod +x cuda_8.0.44_linux-run && \
    ./cuda_8.0.44_linux-run -extract=`pwd`/installers && \
    sudo installers/NVIDIA-Linux-x86_64-367.48.run && \
    modprobe nvidia && \
    sudo installers/cuda-linux64-rel-8.0.44-21122537.run && \
    echo '# CUDA Toolkit' | tee -a .bashrc && \
    echo 'export CUDA_HOME=/usr/local/cuda' | tee -a .bashrc && \
    echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH' | tee -a .bashrc && \
    echo 'export PATH=${CUDA_HOME}/bin:${PATH}' | tee -a .bashrc && \
    source .bashrc

# Installing code dependencies from aptitude
sudo apt-get install -y \
     gfortran \
     libfreetype6-dev \
     libpng12-dev \
     python-pip \
     python2.7 \
     python2.7-dev && \
    echo 'Dependencies from aptitude installed'

# getting code and installing code rependencies from pip
ssh-keygen -f ~/.ssh/id_rsa -t rsa -N '' && \
    echo '~/.ssh/id_rsa.pub:' && cat ~/.ssh/id_rsa.pub && read -p "Copy and add key, enter to continue" && \
    git clone git@bitbucket.org:kidmose/lstm-rnn-correlation.git && \
    LC_ALL=C sudo -H pip install -r lstm-rnn-correlation/requirements.txt `# http://stackoverflow.com/questions/26473681/pip-install-numpy-throws-an-error-ascii-codec-cant-decode-byte-0xe2` && \
    echo 'Requirements installed from pip, code fetched'
