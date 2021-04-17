# Panther

Deep learning based tool to classify protein sequences into Structurally Similar Groups (5 Å distance) based on the CATH database.

# Dataset

The sequences were obtained from the CATH database (Version 4.3.0)

- Total Number of Sequences: 924,212
- Total Number of Classes: 6,396

## Preprocessing

1. Ran an All-vs-all BLAST scan to remove those sequences that had more than 90% sequence similarity but belonged to different superfamilies.
2. Removed those classes that had less than 100 datapoints. 

The resulting dataset has been described below:

- Number of Sequences: 788,287
- Number of Classes: 2,463

The sequences in this dataset were encoded into vector format using a number of techniques and then the performance was compared on each of them. The encoding techniques have been mentioned down below:

- BioVec (Implementation of the ProtVec)
- One-Hot Encoding
- ProtBert 

# Results

## BioVec Embedding Technique

| Model      | Accuracy | Sensitivity | 
| ----------- | ----------- | ----------- |
| CNN   | 96.78%        | 96.68%        |

## One-Hot Encoding

| Model      | Accuracy | Sensitivity | 
| ----------- | ----------- | ----------- |
| CNN   | 98.82%        | 98.76%        |

## ProtBert Embedding

| Model      | Accuracy | Sensitivity | 
| ----------- | ----------- | ----------- |
| CNN   | 99.58%        | 99.57%        |
