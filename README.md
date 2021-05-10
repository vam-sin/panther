# Panther

Deep learning based tool to classify protein sequences into Structurally Similar Groups (5 Å distance) based on the CATH database.

# Dataset

The sequences were obtained from the CATH database (Version 4.3.0)

- Total Number of Sequences: 837,332
- Total Number of Classes: 6,357

## Preprocessing

1. Conducted an all-vs-all BLAST scan to remove cross-hits.
2. Clustered with 80% sequence identity and 80% sequence alignment coverage using MMSeq2.
3. Clustered with S80 representatives with 20% sequence identity and 60% sequence alignment coverage using MMSeq2.
4. Partitioned to test and train.
5. Split into K-Folds.

The resulting training dataset has been described below:

- Number of Sequences: 84,749
- Number of Classes: 2,594

The resulting testing dataset has been described below:

- Number of Sequences: 8,778
- Number of Classes: 2,594

The sequences in this dataset were encoded into vector format using a number of techniques and then the performance was compared on each of them. The encoding techniques have been mentioned down below:

- BioVec (Implementation of the ProtVec)
- One-Hot Encoding
- ProtBert 

# Results

## BioVec Embedding Technique

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.532 ± 0.0047         | 0.524 ± 0.0051         |
| CNN   | 0.364 ± 0.01020         | 0.356 ± 0.01080         |
| CLN   | 0.472 ± 0.0070         | 0.465 ± 0.0065         |
| ABLN   | 0.448 ± 0.0049         | 0.440 ± 0.0044         |

## One-Hot Encoding

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.583 ± 0.0050         | 0.575 ± 0.0046         |
| CNN   | 0.664         | 0.657         |
| CLN   | 0.378         | 0.368         |
| ABLN   | -         | -         |

## ProtBert Embedding

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.852 ± 0.0019         | 0.850 ± 0.0022         |
| CNN   | 0.831 ± 0.0020         | 0.828 ± 0.0022         |
| CLN   | 0.836 ± 0.0007         | 0.834 ± 0.0010         |
| ABLN   | 0.800 ± 0.0024         | 0.797 ± 0.0020         |
