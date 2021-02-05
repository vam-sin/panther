# libraries
import pandas as pd 
import numpy as np 
import pickle

ds = pd.read_csv('../data/dataset.csv')

X = list(ds["sequence"])
y = ds["class"]

chart = ds['class'].value_counts(normalize=True) * 100
# pd.set_option('display.max_rows', None)
print(y)
print(chart)

''' Percentage of each class in the target class: 5515 classes
3.40.50.720_18    0.223416
3.40.640.10_1     0.218989
2.40.30.10_4      0.218965
3.40.50.880_1     0.208394
3.40.50.720_45    0.203556
                    ...   
1.20.1260.10_6    0.000024
2.30.30.240_1     0.000024
3.90.450.1_1      0.000024
3.15.30.10_1      0.000024
1.20.5.630_1      0.000024
Nothing much of significance in the 4,134,000 sequences
'''
