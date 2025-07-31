#!/bin/bash
# applicable to Ascend

MPI_HOME=/usr/local/openmpi
export PATH=${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_with_mpi.sh"
echo "=============================================================================================================="
set -e -x

rm -rf device_pretrain saved_graph
mkdir device_pretrain

cp -r ./datasets ./device_pretrain
cp -r ./checkpoints ./device_pretrain
cp -r ./models ./device_pretrain
cp -r ./lr ./device_pretrain
cp -r ./optim ./device_pretrain
cp -r ./utils ./device_pretrain
cp -r ./*py ./device_pretrain

cd ./device_pretrain

echo "start training"
mpirun python ../pretrain.py