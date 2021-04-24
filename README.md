# Panther

Deep learning based tool to classify protein sequences into Structurally Similar Groups (5 Å distance) based on the CATH database.

# Dataset

The sequences were obtained from the CATH database (Version 4.3.0)

- Total Number of Sequences: 837,332
- Total Number of Classes: 6,357

## Preprocessing

1. Clustered with 95% sequence identity and 80% sequence alignment coverage using MMSeq2.
2. Clustered with S95 representatives with 20% sequence identity and 60% sequence alignment coverage using MMSeq2.
3. Partitioned to test and train.
4. All-vs-all BLAST scan.
5. Split into K-Folds.

The resulting training dataset has been described below:

- Number of Sequences: 235,349
- Number of Classes: 1,707

The resulting testing dataset has been described below:

- Number of Sequences: 12,282
- Number of Classes: 1,707

The sequences in this dataset were encoded into vector format using a number of techniques and then the performance was compared on each of them. The encoding techniques have been mentioned down below:

- BioVec (Implementation of the ProtVec)
- One-Hot Encoding
- ProtBert 

# Results

## BioVec Embedding Technique

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.451 ± 0.0067%        | 0.439 ± 0.0063%        |
| CNN   | 0.395 ± 0.0026%        | 0.385 ± 0.0031%        |

## One-Hot Encoding

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.596 ± 0.0187%        | 0.605 ± 0.0153%        |
| CNN   | 0.699 ± 0.0029%        | 0.697 ± 0.0031%        |

## ProtBert Embedding

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.884 ± 0.0015%        | 0.883 ± 0.0015%        |
| CNN   | 0.874 ± 0.0017%        | 0.872 ± 0.0017%        |
| RPN   | 0.849 ± 0.0006%        | 0.846 ± 0.0004%        |
