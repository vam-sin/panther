#$ -l tmem=24G
#$ -l h_vmem=24G
#$ -l h_rt=72:00:00
#$ -pe smp 4

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N bert_features
#$ -cwd 
source ml-vamsi/bin/activate
source /share/apps/source_files/cuda/cuda-10.1.source
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 protBert_features.py 	