#$ -l tmem=40G
#$ -l gpu=true
#$ -l h_rt=24:00:00

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N pb_train
#$ -cwd 
source ml-actual/bin/activate
cd SSG5/
mkdir PB_Train
source /share/apps/source_files/cuda/cuda-10.1.source
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 ProtBertBFD_Train.py
python3 append_pb_train.py 	