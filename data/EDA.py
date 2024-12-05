"""
Exploratory Data Analysis
"""
## import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.pipeline import make_pipeline

## Exclude life expectancy from output csv?
exclude_life_exp = True

"""
Load datasets
"""
# ## FoodEnvironmentAtlas
# variable_list = pd.read_csv('data/FoodEnvironmentAtlas/VariableList.csv')
# supp_data_state = pd.read_csv('data/FoodEnvironmentAtlas/SupplementalDataState.csv')
# supp_data_county = pd.read_csv('data/FoodEnvironmentAtlas/SupplementalDataCounty.csv')
# state_county_data = pd.read_csv('data/FoodEnvironmentAtlas/StateAndCountyData.csv')
#
# ## HeartDiseaseMortality
# heart_disease_mortality = pd.read_csv('data/HeartDiseaseMortality/Heart_Disease_Mortality.csv')
#
# ## LifeExpectancy
# life_expectancy = pd.read_csv('data/LifeExpectancy/U.S._Life_Expectancy.csv')

"""
Objective 1 feature selection
"""
## List of features to keep from state_county_data
state_county_features = ["PCT_DIABETES_ADULTS13", "PCT_65OLDER10", "PCT_18YOUNGER10", "MEDHHINC15", "POVRATE15", "METRO13", "PCT_LACCESS_POP15", "PCT_LACCESS_LOWI15", "PCT_LACCESS_HHNV15", "PCT_LACCESS_SNAP15", "PCT_LACCESS_CHILD15", "PCT_LACCESS_SENIORS15", "GROC11", "SUPERC11", "CONVS11", "SPECS11", "SNAPSPTH12", "PC_SNAPBEN12", "FFR11", "FSR11", "SODATAX_STORES14", "SODATAX_VENDM14", "CHIPSTAX_STORES14", "CHIPSTAX_VENDM14", "FOOD_TAX14", "DIRSALES_FARMS12", "DIRSALES12", "PC_DIRSALES12", "FMRKT13", "FMRKT_SNAP13", "PCT_FMRKT_FRVEG13", "PCT_FMRKT_ANMLPROD13", "VEG_FARMS12", "VEG_ACRES12", "ORCHARD_FARMS12", "ORCHARD_ACRES12", "BERRY_FARMS12", "BERRY_ACRES12", "SLHOUSE12", "GHVEG_FARMS12", "CSA12", "AGRITRSM_OPS12", "AGRITRSM_RCT12", "RECFAC11", "PCT_NHWHITE10", "PCT_NHBLACK10", "PCT_HISP10"]

## Select features and tidy dataframe
state_county_data = pd.read_csv('data/FoodEnvironmentAtlas/StateAndCountyData.csv')
state_county_data['CountySt'] = state_county_data['County'].str.cat(state_county_data['State'], sep=', ')
df = state_county_data[state_county_data["Variable_Code"].isin(state_county_features)].pivot_table(index="CountySt", columns="Variable_Code", values="Value", aggfunc='max')
del state_county_data

## Merge supplementary data
supp_data_county = pd.read_csv('data/FoodEnvironmentAtlas/SupplementalDataCounty.csv')
supp_data_county['County'] = [c if 'County' not in c else c[:-7] for c in supp_data_county['County']]
supp_data_county['CountySt'] = supp_data_county['County'].str.cat(supp_data_county['State'], sep=',')
supp_data_county = supp_data_county.pivot(index="CountySt", columns="Variable_Code", values="Value")

df = df.join(supp_data_county, how="left")
df.drop(supp_data_county.columns[1:], axis=1, inplace=True)

## Merge heart disease mortality
hdm = pd.read_csv('data/HeartDiseaseMortality/Heart_Disease_Mortality.csv')
hdm = hdm.loc[(hdm["Stratification1"]=="Overall") & (hdm["Stratification2"]=="Overall"), ["LocationAbbr", "LocationDesc", "Data_Value"]]
hdm.rename(columns={'LocationAbbr': 'State', 'LocationDesc': 'County', 'Data_Value': 'HDM'}, inplace=True)
hdm['County'] = [c if 'County' not in c else c[:-7] for c in hdm['County']]
hdm['CountySt'] = hdm['County'].str.cat(hdm['State'], sep=', ')
hdm.drop(['State', 'County'], axis=1, inplace=True)
hdm.set_index('CountySt', inplace=True)
df = df.join(hdm, how='left')

## Merge life expectancy
life_expectancy = pd.read_csv('data/LifeExpectancy/U.S._Life_Expectancy.csv')
life_expectancy.dropna(axis=0, how='any', inplace=True)
life_expectancy['County'] = life_expectancy['County'].str.replace(' County', '')
life_expectancy.rename(columns={'County': 'CountySt'}, inplace=True)
life_expectancy = life_expectancy[['CountySt','Life Expectancy']].groupby(['CountySt']).mean()
df = df.join(life_expectancy, how='left')

## Remove duplicated indices
df = df.groupby(level=0).max()

# ## Compare response variables
# sns.pairplot(df, vars=['HDM', 'Life Expectancy', 'PCT_DIABETES_ADULTS13'], corner=True, diag_kind='kde', kind='kde')
# plt.title('Response variable pairplot kde')
# plt.savefig('data/figures/Response variable pairplot.png')
# plt.show()

# sns.heatmap(df[['HDM', 'Life Expectancy', 'PCT_DIABETES_ADULTS13']].corr(), annot=True)
# plt.title('Response variable correlations')
# plt.savefig('data/figures/Response variable corr heatmap.png')
# plt.show()

## Response variable PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# choose response vars to use
resp_vars = ['HDM', 'PCT_DIABETES_ADULTS13']

# select response variables and standardize
pca_df = df[resp_vars].copy()
pca_df.dropna(axis=0, how='any', inplace=True)
pca_df[resp_vars] = StandardScaler().fit_transform(pca_df)

# PCA fit, transform, calculate explained variance
explained_var = PCA(random_state=10).fit(pca_df).explained_variance_ratio_
pca_df['PCA'] = PCA(random_state=10, n_components=1).fit_transform(pca_df)

# # compare first principle component with raw response variables
# sns.pairplot(pca_df, corner=True, diag_kind='kde', kind='kde')
# plt.title('Response variable PCA pairplot kde')
# plt.savefig('data/figures/Response variable PCA pairplot.png')
# plt.show()

# # display correlations between response variables
# plt.figure(figsize=(10,8))
# sns.heatmap(pca_df.corr(), annot=True)
# plt.title('Response variable PCA correlations')
# plt.savefig('data/figures/Response variable PCA corr heatmap.png')
# plt.show()

# count nans in response variables
df[resp_vars].isna().sum()
print(f'% data retained after removing response variable nans: {pca_df.shape[0] / df.shape[0]*100:.2f}%\n')

# merge PCA response variable
df = df.join(pca_df['PCA'], how='left')

## Per capita variables
per_cap_vars = ['FFR11', 'FSR11', 'GROC11', 'SPECS11', 'RECFAC11', 'CONVS11', 'SUPERC11', 'FMRKT13']
df[per_cap_vars] = df[per_cap_vars].add(1).div(df['2010_Census_Population'], axis=0)

## Collinearity
corr = df.corr().abs().unstack()
corr = corr.sort_values(ascending=False)
print('__________\nCollinear features\n__________')
print('\n'.join([f'{c1}, {c2}: {corr[c1,c2]:.4f}' for c1, c2 in corr.index if c1 != c2 and corr[c1,c2] > 0.7]))

# drop collinear variables
df.drop(['CHIPSTAX_STORES14', 'PCT_LACCESS_CHILD15', 'PCT_LACCESS_SENIORS15', 'PCT_LACCESS_LOWI15', 'PC_SNAPBEN12'], axis=1, inplace=True)

## Nan analysis
# count nans in all variables
nan_count = df.isna().sum()
print('__________\nFeatures with >5% nans\n__________\n')
print(nan_count[nan_count > 0.05*df.shape[0]])

# drop variables with >5% nans (except census population and life expectancy)
df.drop(['AGRITRSM_RCT12', 'BERRY_ACRES12', 'DIRSALES12', 'ORCHARD_ACRES12', 'PC_DIRSALES12', 'VEG_ACRES12'], axis=1, inplace=True)

# % counties with nans
print(f'counties without nans = {df.dropna(axis=0).shape[0]}, {df.dropna(axis=0).shape[0]/df.shape[0]*100:.2f}%')

## Check if counties with nans have worse health
nan_count_per_row = df.isna().sum(axis=1)
nan_df = pd.DataFrame(nan_count_per_row).join(df['PCT_DIABETES_ADULTS13'], how='left')
nan_df.rename(columns={0: 'nans'}, inplace=True)
nan_df['has_nans'] = nan_df['nans'] > 0

# # boxplot
# sns.boxplot(x=nan_df['PCT_DIABETES_ADULTS13'], hue=nan_df['has_nans'])
# plt.title('Difference in diabetes rates for counties with nans')
# plt.savefig('data/figures/diabetes rates vs nans boxplot.png')
# plt.show()

# t test
import scipy.stats as stats
ttest = stats.ttest_ind(nan_df.loc[nan_df['has_nans']==True, 'PCT_DIABETES_ADULTS13'], nan_df.loc[nan_df['has_nans']==False, 'PCT_DIABETES_ADULTS13'])
print(f'counties with nans have diabetes rates between {ttest.confidence_interval()[0]:.4f} and {ttest.confidence_interval()[1]:.4f} greater than counties without nans, p-value = {ttest[1]:.4e}')

# remove life expectancy before dropping nans?
if exclude_life_exp:
    df.drop('Life Expectancy', axis=1, inplace=True)

# remove rows with nans
df.dropna(axis=0, inplace=True)

# stores PCA
from sklearn.pipeline import make_pipeline
stores_vars = ['SUPERC11', 'RECFAC11', 'SPECS11', 'FMRKT13', 'GROC11']
df[stores_vars] = np.log(df[stores_vars])
df['storesPCA'] = make_pipeline(StandardScaler(), PCA(n_components=1)).fit_transform(df[stores_vars])
sns.pairplot(df, x_vars=stores_vars, y_vars='storesPCA')
plt.show()
stores_pca = make_pipeline(StandardScaler(), PCA()).fit(df[stores_vars])
print(f'Stores PCA explained variance: \n{stores_pca[-1].explained_variance_ratio_}')
df.drop(stores_vars, axis=1, inplace=True)

# ## Nonlinearity
# yvars = ['PCT_DIABETES_ADULTS13', 'HDM', 'PCA']
# if not exclude_life_exp:
#     yvars += ['Life Expectancy']
# xvars = [v for v in df.columns if v not in yvars]
# sns.pairplot(df, x_vars=xvars, y_vars=yvars, diag_kind='kde')
# plt.title('Response variables vs features pairplot (check for nonlinearity)')
# plt.savefig('data/figures/nonlinearity check pairplot.png')
# plt.show()

## Check normality of all variables
fig, axs = plt.subplots(7, 6, figsize=(15, 15))
axs = axs.flatten()
for i,col in enumerate(df.columns):
    sns.kdeplot(df[col], ax=axs[i])
plt.tight_layout()
plt.suptitle('Check distributions of all features')
# plt.savefig('data/figures/feature normality check kde.png')
plt.show()

# count zeros for each feature
print(f'count 0s per feature:\n{(df == 0).sum(axis=0)}')

# remove bimodal features
bimodal_vars = ['CHIPSTAX_VENDM14', 'FOOD_TAX14', 'SODATAX_STORES14', 'SODATAX_VENDM14']
df.drop(bimodal_vars, axis=1, inplace=True)

# test for normality
from scipy.stats import shapiro
import numpy as np
res = shapiro(df, axis=0)
normality_test = pd.DataFrame({'Features': df.columns, 'Statistic': res.statistic, 'pvalue': res.pvalue})

# log transform
log_0_vars = ['MEDHHINC15', 'PCT_18YOUNGER10', 'PCT_65OLDER10', 'POVRATE15', '2010_Census_Population', 'HDM', 'PCT_DIABETES_ADULTS13']
if not exclude_life_exp:
    log_0_vars += ['Life Expectancy']
log_1_vars = ['ORCHARD_FARMS12', 'BERRY_FARMS12', 'GHVEG_FARMS12', 'VEG_FARMS12', 'SLHOUSE12', 'CSA12', 'AGRITRSM_OPS12', 'DIRSALES_FARMS12', 'SNAPSPTH12']
log_01_vars = ['PCT_LACCESS_HHNV15', 'PCT_LACCESS_POP15', 'PCT_LACCESS_SNAP15']
print(f'Check overlapping vars: {set(log_0_vars) & set(log_1_vars) & set(log_01_vars)}\nMissing vars: {set(df.columns) - set(log_0_vars) - set(log_1_vars) - set(log_01_vars)}')

df_raw = df.copy()
df[log_0_vars] = df[log_0_vars].transform(lambda x: np.log(x))
df[log_1_vars] = df[log_1_vars].transform(lambda x: np.log(x+1))
df[log_01_vars] = df[log_01_vars].transform(lambda x: np.log(x+0.1))
from sklearn.preprocessing import QuantileTransformer
quant_trans_vars = ['CONVS11', 'FFR11', 'FSR11', 'PCT_NHWHITE10', 'PCT_HISP10','PCT_NHBLACK10']
df[quant_trans_vars] = QuantileTransformer(random_state=10, output_distribution='normal').fit_transform(df.loc[:, quant_trans_vars])

# retest normality
fig, axs = plt.subplots(6, 5, figsize=(15, 15))
axs = axs.flatten()
for i,col in enumerate(df.columns):
    sns.kdeplot(df[col], ax=axs[i])
plt.tight_layout()
plt.suptitle('Recheck distributions of all features after log-transform')
plt.savefig('data/figures/feature normality check kde log transformed.png')
plt.show()

res = shapiro(df, axis=0)
normality_test['Log_statistic'] = res.statistic
normality_test['Log_pvalue'] = res.pvalue

# create dummy vars
df_transformed = df.copy()
dummy_vars = ['SLHOUSE12', 'GHVEG_FARMS12', 'CSA12', 'BERRY_FARMS12']
for dv in dummy_vars:
    df.loc[df[dv] > 0, dv] = 1.0

# final distribution check
fig, axs = plt.subplots(6, 5, figsize=(15, 15))
axs = axs.flatten()
for i,col in enumerate(df.columns):
    sns.kdeplot(df[col], ax=axs[i])
plt.tight_layout()
plt.suptitle('Recheck distributions of all features after log-transform and dummy vars')
plt.savefig('data/figures/final distribution check.png')
plt.show()

"""
Detrending data by income
"""
# model income vs health
from sklearn.linear_model import LinearRegression
X, y = df.loc[:,['MEDHHINC15']], df['PCA']
reg = LinearRegression().fit(X=X, y=y)
coef = reg.coef_
pred = reg.predict(X=X)
res = y - pred
r2 = reg.score(X=X, y=y)

# check normality
fig, axs = plt.subplots(1, 2, figsize=(8, 5))
sns.histplot(res, ax=axs[0])
sns.regplot(x=pred, y=res, ax=axs[1], scatter_kws={'alpha':0.4}, line_kws=dict(color="r"))
axs[0].set_xlabel('Residuals')
axs[1].set_xlabel('Predictions')
axs[1].set_ylabel('Residuals')
fig.suptitle(f'Income trend normality check (coef = {coef[0]:.3f}, R2 = {r2:.3f})')
plt.savefig('data/figures/income trend normality check.png')
plt.show()

# record detrended data
df['PCA_dt_medinc'] = res

## Detrend by all income vars
# PCA of income vars
income_vars = ['MEDHHINC15', 'PCT_LACCESS_HHNV15', 'POVRATE15', 'SNAPSPTH12']
income_PCA = df[income_vars].copy()
income_PCA['PCA'] = make_pipeline(StandardScaler(), PCA(n_components=1)).fit_transform(income_PCA)

sns.pairplot(income_PCA, x_vars=income_vars, y_vars=['PCA'], diag_kind='kde')
plt.show()

# model income vars PCA vs health
X, y = income_PCA.loc[:,['PCA']], df['PCA']
reg = LinearRegression().fit(X=X, y=y)
coef = reg.coef_
pred = reg.predict(X=X)
res = y - pred
r2 = reg.score(X=X, y=y)

# check normality
fig, axs = plt.subplots(1, 2, figsize=(8, 5))
sns.histplot(res, ax=axs[0])
sns.regplot(x=pred, y=res, ax=axs[1], scatter_kws={'alpha':0.4}, line_kws=dict(color="r"))
axs[0].set_xlabel('Residuals')
axs[1].set_xlabel('Predictions')
axs[1].set_ylabel('Residuals')
fig.suptitle(f'Income PCA trend normality check (coef = {coef[0]:.3f}, R2 = {r2:.3f})')
plt.savefig('data/figures/income PCA trend normality check.png')
plt.show()

# record detrended data
df['PCA_dt_allinc'] = res

"""
Export datasets
"""
## Write objective 1 dataset to csv
df.to_csv('./data/objective1.csv', index_label='CountySt')
df_raw.to_csv('./data/objective1_raw.csv', index_label='CountySt')