"""
Objective 1: Model the relationship between health factors and socioeconomic factors, store access, restaurant availability, and farms and markets. Which factors are the best predictors of health?

Method: LASSO regression + CV
"""
## Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Read data
df = pd.read_csv('./data/objective1.csv', index_col='CountySt')