#$ -l tmem=128G
#$ -l h_rt=24:00:00
#$ -l m_core=8
#$ -pe smp 4

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N rm_xhit
#$ -cwd 
source ml-actual/bin/activate
cd SSG5/
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 remove_xhit.py 	