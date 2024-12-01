"""
Exploratory Data Analysis
"""
## import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
state_county_features = ["PCT_DIABETES_ADULTS13", "PCT_65OLDER10", "PCT_18YOUNGER10", "MEDHHINC15", "POVRATE15", "METRO13", "PCT_LACCESS_POP15", "PCT_LACCESS_LOWI15", "PCT_LACCESS_HHNV15", "PCT_LACCESS_SNAP15", "PCT_LACCESS_CHILD15", "PCT_LACCESS_SENIORS15", "GROCPTH11", "SUPERCPTH11", "CONVSPTH11", "SPECSPTH11", "SNAPSPTH12", "PC_SNAPBEN12", "FFRPTH11", "FSRPTH11", "SODATAX_STORES14", "SODATAX_VENDM14", "CHIPSTAX_STORES14", "CHIPSTAX_VENDM14", "FOOD_TAX14", "DIRSALES_FARMS12", "DIRSALES12", "PC_DIRSALES12", "FMRKT13", "FMRKTPTH13", "FMRKT_SNAP13", "PCT_FMRKT_FRVEG13", "PCT_FMRKT_ANMLPROD13", "VEG_FARMS12", "VEG_ACRES12", "ORCHARD_FARMS12", "ORCHARD_ACRES12", "BERRY_FARMS12", "BERRY_ACRES12", "SLHOUSE12", "GHVEG_FARMS12", "CSA12", "AGRITRSM_OPS12", "AGRITRSM_RCT12", "RECFACPTH11"]

## Select features and tidy dataframe
state_county_data = pd.read_csv('data/FoodEnvironmentAtlas/StateAndCountyData.csv')
state_county_data['CountySt'] = state_county_data['County'].str.cat(state_county_data['State'], sep=', ')
obj1_df = state_county_data[state_county_data["Variable_Code"].isin(state_county_features)].pivot_table(index="CountySt", columns="Variable_Code", values="Value", aggfunc='max')

## Merge supplementary data
supp_data_county = pd.read_csv('data/FoodEnvironmentAtlas/SupplementalDataCounty.csv')
supp_data_county['County'] = [c if 'County' not in c else c[:-7] for c in supp_data_county['County']]
supp_data_county['CountySt'] = supp_data_county['County'].str.cat(supp_data_county['State'], sep=',')
supp_data_county = supp_data_county.pivot(index="CountySt", columns="Variable_Code", values="Value")

obj1_df = obj1_df.join(supp_data_county, how="left")
obj1_df.drop(supp_data_county.columns[1:], axis=1, inplace=True)

## Merge heart disease mortality
hdm = pd.read_csv('data/HeartDiseaseMortality/Heart_Disease_Mortality.csv')
hdm = hdm.loc[(hdm["Stratification1"]=="Overall") & (hdm["Stratification2"]=="Overall"), ["LocationAbbr", "LocationDesc", "Data_Value"]]
hdm.rename(columns={'LocationAbbr': 'State', 'LocationDesc': 'County', 'Data_Value': 'HDM'}, inplace=True)
hdm['County'] = [c if 'County' not in c else c[:-7] for c in hdm['County']]
hdm['CountySt'] = hdm['County'].str.cat(hdm['State'], sep=', ')
hdm.drop(['State', 'County'], axis=1, inplace=True)
hdm.set_index('CountySt', inplace=True)
obj1_df = obj1_df.join(hdm, how='left')

## Merge life expectancy
life_expectancy = pd.read_csv('data/LifeExpectancy/U.S._Life_Expectancy.csv')
life_expectancy.dropna(axis=0, how='any', inplace=True)
life_expectancy['County'] = life_expectancy['County'].str.replace(' County', '')
life_expectancy.rename(columns={'County': 'CountySt'}, inplace=True)
life_expectancy = life_expectancy[['CountySt','Life Expectancy']].groupby(['CountySt']).mean()
obj1_df = obj1_df.join(life_expectancy, how='left')

## Remove duplicated indices
obj1_df = obj1_df.groupby(level=0).max()

## Compare response variables
sns.pairplot(obj1_df, vars=['HDM', 'Life Expectancy', 'PCT_DIABETES_ADULTS13'], corner=True, diag_kind='kde', kind='kde')
plt.show()

sns.heatmap(obj1_df[['HDM', 'Life Expectancy', 'PCT_DIABETES_ADULTS13']].corr(), annot=True)
plt.show()

## Response variable PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# select response variables and standardize
pca_df = obj1_df[['HDM', 'Life Expectancy', 'PCT_DIABETES_ADULTS13']].copy()
pca_df.dropna(axis=0, how='any', inplace=True)
pca_df[['HDM', 'Life Expectancy', 'PCT_DIABETES_ADULTS13']] = StandardScaler().fit_transform(pca_df)

# PCA fit, transform, calculate explained variance
explained_var = PCA(random_state=10).fit(pca_df).explained_variance_ratio_
pca_df['PCA'] = PCA(random_state=10, n_components=1).fit_transform(pca_df)

# compare first principle component with raw response variables
sns.pairplot(pca_df, corner=True, diag_kind='kde', kind='kde')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(pca_df.corr(), annot=True)
plt.show()

# count nans in response variables
obj1_df[['HDM', 'Life Expectancy', 'PCT_DIABETES_ADULTS13']].isna().sum()
print(f'% data retained after removing response variable nans: {pca_df.shape[0] / obj1_df.shape[0]*100:.2f}')

# merge PCA response variable
obj1_df = obj1_df.join(pca_df['PCA'], how='left')

## Collinearity
corr = obj1_df.corr().abs().unstack()
corr = corr.sort_values(ascending=False)
print('\n'.join([f'{c1}, {c2}: {corr[c1,c2]:.4f}' for c1, c2 in corr.index if c1 != c2 and corr[c1,c2] > 0.7]))

# drop collinear variables
obj1_df.drop(['CHIPSTAX_STORES14', 'PCT_LACCESS_CHILD15', 'PCT_LACCESS_SENIORS15', 'PCT_LACCESS_LOWI15', 'PC_SNAPBEN12'], axis=1, inplace=True)

## Nan analysis
# count nans in all variables
nan_count = obj1_df.isna().sum()
print(nan_count[nan_count > 0.05*obj1_df.shape[0]])

# drop variables with >5% nans (except census population and life expectancy)
obj1_df.drop(['AGRITRSM_RCT12', 'BERRY_ACRES12', 'DIRSALES12', 'ORCHARD_ACRES12', 'PC_DIRSALES12', 'VEG_ACRES12'], axis=1, inplace=True)

# % counties with nans
print(f'counties without nans = {obj1_df.dropna(axis=0).shape[0]}, {obj1_df.dropna(axis=0).shape[0]/obj1_df.shape[0]*100:.2f}%')

## Check if counties with nans have worse health
nan_count_per_row = obj1_df.isna().sum(axis=1)
nan_df = pd.DataFrame(nan_count_per_row).join(obj1_df['PCT_DIABETES_ADULTS13'], how='left')
nan_df.rename(columns={0: 'nans'}, inplace=True)
nan_df['has_nans'] = nan_df['nans'] > 0

# boxplot
sns.boxplot(x=nan_df['PCT_DIABETES_ADULTS13'], hue=nan_df['has_nans'])
plt.show()

# t test
import scipy.stats as stats
ttest = stats.ttest_ind(nan_df.loc[nan_df['has_nans']==True, 'PCT_DIABETES_ADULTS13'], nan_df.loc[nan_df['has_nans']==False, 'PCT_DIABETES_ADULTS13'])
print(f'counties with nans have diabetes rates between {ttest.confidence_interval()[0]:.4f} and {ttest.confidence_interval()[1]:.4f} greater than counties without nans, p-value = {ttest[1]:.4e}')

# remove rows with nans
obj1_df.dropna(axis=0, inplace=True)

## Nonlinearity
yvars = ['PCT_DIABETES_ADULTS13', 'HDM', 'Life Expectancy', 'PCA']
xvars = [v for v in obj1_df.columns if v not in yvars]
sns.pairplot(obj1_df, x_vars=xvars, y_vars=yvars, diag_kind='kde')
plt.show()

## Write objective 1 dataset to csv
obj1_df.to_csv('./data/objective1.csv', index_label='CountySt')