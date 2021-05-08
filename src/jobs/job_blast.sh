#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=24:00:00
#$ -l m_core=48
#$ -pe smp 4

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N blast_xhit
#$ -cwd 
cd SSG5/
ncbi-blast-2.11.0+/bin/blastp -db SSG5_db -query SSG5.fasta -outfmt 6 -out all-vs-all.tsv -num_threads 48