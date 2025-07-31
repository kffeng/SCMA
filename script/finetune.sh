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

rm -rf device_finetune saved_graph
mkdir device_finetune

cp -r ./datasets ./device_finetune
cp -r ./checkpoints ./device_finetune
cp -r ./models ./device_finetune
cp -r ./lr ./device_finetune
cp -r ./optim ./device_finetune
cp -r ./utils ./device_finetune
cp -r ./*py ./device_finetune

cd ./device_finetune

echo "start training"
mpirun python ./finetune.py