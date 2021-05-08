#$ -l tmem=10G
#$ -l gpu=true
#$ -l h_rt=24:00:00

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N cnn_protbert
#$ -cwd 
source ml-actual/bin/activate
cd SSG5/
source /share/apps/source_files/cuda/cuda-10.1.source
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 cnn_protbert.py