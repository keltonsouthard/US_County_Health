"""
Exploratory Data Analysis
"""
## import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Load datasets
"""
## FoodEnvironmentAtlas
variable_list = pd.read_csv('data/FoodEnvironmentAtlas/VariableList.csv')
supp_data_state = pd.read_csv('data/FoodEnvironmentAtlas/SupplementalDataState.csv')
supp_data_county = pd.read_csv('data/FoodEnvironmentAtlas/SupplementalDataCounty.csv')
state_county_data = pd.read_csv('data/FoodEnvironmentAtlas/StateAndCountyData.csv')

## HeartDiseaseMortality
heart_disease_mortality = pd.read_csv('data/HeartDiseaseMortality/Heart_Disease_Mortality.csv')

## LifeExpectancy
life_expectancy = pd.read_csv('data/LifeExpectancy/U.S._Life_Expectancy.csv')