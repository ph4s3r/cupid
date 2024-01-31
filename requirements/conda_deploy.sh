#!/bin/bash

##############################################
# Installation sequence with conda on debian #
# Usage: sudo bash conda_deploy.sh
##############################################


### NVIDIA drivers (debian)
### https://www.linuxcapable.com/install-nvidia-drivers-on-debian/


### install conda to /opt/anaconda folder
### https://docs.anaconda.com/free/anaconda/install/linux/
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
sudo curl -fsSL https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -o /opt/Anaconda3-2023.09-0-Linux-x86_64.sh
sudo bash /opt/Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda || true
PATH=$(echo $PATH:/opt/anaconda/bin/)
alias "conda=/opt/anaconda/bin/conda" >> ~/.bashrc
conda init bash


### restart bash
ScriptLoc=$(readlink -f "$0")
chmod 700 $ScriptLoc
exec "$ScriptLoc"


### creat conda env:
conda create -n pathml python=3.9 -y
echo "conda activate pathml" >> ~/.bashrc
conda config --set auto_activate_base false


### install conda packages
conda install -y -q pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# if cuda was not installed by dpkg/apt then conda install nvidia/label/cuda-12.1.1::cuda


### pip
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
pip install coolname scikit_learn pandas
sudo apt-get install openslide-tools g++ gcc libblas-dev liblapack-dev # pathml deps
conda install -y openjdk==8.0.152
pip install pathml==2.1.1
pip install ray==2.9.1 # do not install ray with conda!!! buggy because train.Checkpoint is not found
pip install "ray[tune]"
pip install hyperopt
# pathml had a GLIBC error where it does not found a library, so need to link the 
# ln -sf /home/peet/.conda/envs/pathml/lib/libstdc++.so.6 /home/peet/.conda/envs/pathml/lib/libstdc++.so.6