#!/bin/bash

#PBS -N model2
#PBS -o model2.out
#PBS -e model2.err
#PBS -l select=ncpus=8:ngpus=1:mem=46G
#PBS -l walltime=12:00:00
#PBS -M z5114185@ad.unsw.edu.au
#PBS -m ae

cd /home/z5114185/thesis/undergrad/width/model2
rm -r pyEnv
mkdir pyEnv
module load python/3.7.3
python3 -m venv --system-site-packages pyEnv
. pyEnv/bin/activate
pip3 install tensorflow-gpu==2.0.0
python3 train.py  /srv/scratch/z5114185/thesis/datasetV1.1/training_set/ > model2.log
deactivate
rm -r pyEnv
