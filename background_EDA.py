"""
Background EDA
Use our dataset to explore trends/correlations found in other studies regarding public health
"""
## Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## Load data
df = pd.read_csv('data/objective1.csv', index_col='CountySt')

## Invert health metric?
neg_resp_var = True
if neg_resp_var:
    df['PCA'] *= -1

## Convert dummy vars to character
dummy_vars = ['METRO13']
chr_df = df.copy()
for dv in dummy_vars:
    chr_df[dv] = ['Yes' if x==1 else 'No' for x in chr_df[dv]]

"""
How Healthy Is Your County?
https://pmc.ncbi.nlm.nih.gov/articles/PMC2935645/#:~:text=Each%20county%20is%20rated%20on,of%20children%20living%20in%20poverty.

1. "80% of the counties with populations in poorest health were rural" 
Trapp D.Health status varies by county: Where patients live matters American Medical News March12010. Available at: www.ama-assn.org/amednews/2010/03/01/gvl10201.htm Accessed June 30, 2010.

2. Conversely, people who live in the healthier-ranked counties tend to have higher education levels, are more likely to be employed; and have access to more health care providers, healthful foods, parks, and recreational facilities.
"""

## Question 1: Do rural counties have worse health? Do urban or rural counties tend to have larger populations?
fig, axs = plt.subplots(1,2, figsize=(8,5))
sns.boxplot(data=chr_df, x='PCA', hue='METRO13', ax=axs[0])
sns.boxplot(data=chr_df, x=np.log10(chr_df['2010_Census_Population']), hue='METRO13', ax=axs[1])
axs[0].set_ylabel('Urban vs Rural')
axs[0].set_xlabel('Health score')
axs[1].set_xlabel('log10 County population')
axs[0].legend(title='Urban?')
axs[1].legend(title='Urban?')
plt.suptitle('Urban counties tend to be larger and slightly healthier than rural counties')
plt.savefig('data/figures/urban vs rural.png')
plt.show()

## Question 2: Do healthier counties have greater access to healthful foods and recreational facilities?
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

healthy_farm_cols = ['AGRITRSM_OPS12', 'BERRY_FARMS12', 'CSA12', 'FMRKT13', 'GHVEG_FARMS12', 'ORCHARD_FARMS12', 'VEG_FARMS12']
healthy_store_cols = ['FSR11', 'GROC11', 'SPECS11']
unhealthy_store_cols = ['CONVS11', 'FFR11']

healthy_farms = df[healthy_farm_cols].copy()
healthy_stores = df[healthy_store_cols].copy()
unhealthy_stores = df[unhealthy_store_cols].copy()

# PCA
healthy_farms['PCA'] = make_pipeline(StandardScaler(), PCA(n_components=1)).fit_transform(healthy_farms)
healthy_stores['PCA'] = make_pipeline(StandardScaler(), PCA(n_components=1)).fit_transform(healthy_stores)
unhealthy_stores['PCA'] = make_pipeline(StandardScaler(), PCA(n_components=1)).fit_transform(unhealthy_stores)

# PCA pairplot
df_list = [healthy_farms, healthy_stores, unhealthy_stores]
for d in df_list:
    sns.pairplot(d, x_vars=d.columns[:-1], y_vars=['PCA'], kind='scatter')
    plt.show()

pca_df = pd.DataFrame({'farm access': healthy_farms['PCA'], 'health stores': healthy_stores['PCA'], 'unhealthy stores': unhealthy_stores['PCA']})
sns.pairplot(pca_df, corner=True, diag_kind='kde', kind='kde')
plt.show()

# plot food vs health
fig, axs = plt.subplots(1,3, figsize=(10,4))
sns.regplot(x=healthy_farms['PCA'], y=df['PCA'], ax=axs[0], scatter_kws={'alpha': 0.2}, line_kws=dict(color="r"))
axs[0].set_ylabel('Health score')
axs[0].set_xlabel('Farm access')
sns.regplot(x=healthy_stores['PCA'], y=df['PCA'], ax=axs[1], scatter_kws={'alpha': 0.2}, line_kws=dict(color="r"))
axs[1].set_ylabel('')
axs[1].set_xlabel('Healthy store access')
sns.regplot(x=unhealthy_stores['PCA'], y=df['PCA'], ax=axs[2], scatter_kws={'alpha': 0.2}, line_kws=dict(color="r"))
axs[2].set_ylabel('')
axs[2].set_xlabel('Unhealthy store access')
plt.suptitle('Health improves with all food access,\nbut access to full service restaurants, grocery stores, and specialty stores ("healthy stores") has the largest correlation')
plt.savefig('data/figures/healthy food access.png')
plt.show()

## Question 3: Are wealthier counties healthier?
fig, axs = plt.subplots(1,2, figsize=(8,5))
sns.regplot(data=df, x='MEDHHINC15', y='PCA', ax=axs[0], scatter_kws={'alpha': 0.2}, line_kws=dict(color="r"))
axs[0].set_ylabel('Health score')
axs[0].set_xlabel('Median household income')
sns.regplot(data=df, x='POVRATE15', y='PCA', ax=axs[1], scatter_kws={'alpha': 0.2}, line_kws=dict(color="r"))
axs[1].set_ylabel('')
axs[1].set_xlabel('Poverty rate')
plt.suptitle('Health increases with income and declines with poverty rate')
plt.savefig('data/figures/health vs income.png')
plt.show()

