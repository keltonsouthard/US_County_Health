"""
Objective 1: Model the relationship between health factors and socioeconomic factors, store access, restaurant availability, and farms and markets. Which factors are the best predictors of health?

Method: LASSO regression + CV
"""
## Reverse response var sign?
neg_resp_var = True

## Set directory
import os
import git

def get_git_root(path):
    """Find the root of a Git repository."""
    try:
        repo = git.Repo(path, search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        return None

# Get the root of the Git repository
git_root = get_git_root(os.getcwd())

if git_root:
    # Change the working directory to the Git root
    os.chdir(git_root)
    print("Working directory changed to:", os.getcwd())
else:
    print("Not in a Git repository")

## Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

## Read data
df = pd.read_csv('./data/objective1.csv', index_col='CountySt')

## Response variable selection
select_var = 'PCA'
resp_vars = ['PCT_DIABETES_ADULTS13', 'HDM', 'PCA']
df.drop([v for v in resp_vars if v != select_var], axis=1, inplace=True)
df[select_var] *= -1 if neg_resp_var else 1

## X, y
y = df[select_var]
x_cols = [c for c in df.columns if c not in resp_vars]
X = df[x_cols]

## Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

"""
LASSO model sklearn
"""
def lasso_model(model, X_train, y_train, plot=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ## Fit model (from https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py)
    lasso_pipe = make_pipeline(StandardScaler(), model)
    lasso_pipe.fit(X_train, y_train)
    lasso = lasso_pipe[-1]

    if plot:
        # plot
        plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
        plt.plot(lasso.alphas_, lasso.mse_path_.mean(axis=-1), color="black", label="Average across the folds", linewidth=2)
        plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean square error")
        plt.legend()
        plt.title(f"Mean square error on each fold: coordinate descent")
        plt.savefig('data/figures/lassocv coordinate descent')
        plt.show()

    # print training results
    print(f'________ \n{lasso} \n________')
    try:
        alpha = lasso.alpha_
    except:
        alpha = lasso.alpha
    print(f'best alpha: {alpha}')
    coefs = pd.DataFrame({'feature': X_train.columns, 'coefficients': lasso.coef_})
    print(f'top coefficients: \n{coefs.sort_values(ascending=False, by='coefficients').head()}')
    print(f'# zero coefs = {coefs[coefs['coefficients']==0].shape[0]}, # nonzero coefs = {coefs[coefs['coefficients']!=0].shape[0]} \n{coefs.loc[coefs['coefficients'] == 0, 'feature']}')
    print(f'R2: {lasso_pipe.score(X_train, y_train):.4f}')

    return lasso_pipe, lasso

## Fit models
lassocv_pipe, lassocv = lasso_model(LassoCV(random_state=10), X_train, y_train, plot=True) # (from https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py)
lasso_pipe, lasso = lasso_model(Lasso(alpha=0.1, random_state=10), X_train, y_train)
lassolarsaic_pipe, lassolarsaic = lasso_model(LassoLarsIC(criterion='aic'), X_train, y_train)
lassolarsbic_pipe, lassolarsbic = lasso_model(LassoLarsIC(criterion='bic'), X_train, y_train)

# alpha vs number of features
alphas = np.logspace(-3, 0, 50)
alpha_features = pd.DataFrame({'alpha':[], 'R2':[], 'features':[]})
for alpha in alphas:
    lasso_pipe = make_pipeline(StandardScaler(), Lasso(alpha=alpha, random_state=10))
    lasso_pipe.fit(X_train, y_train)
    coefs = lasso_pipe[-1].coef_
    tempdf = pd.DataFrame({'alpha': [alpha], 'R2': [lasso_pipe.score(X_train, y_train)], 'features': [np.count_nonzero(coefs)]})
    alpha_features = pd.concat([alpha_features, tempdf], axis=0)

# plot R2 vs features
plt.scatter(alpha_features['features'], alpha_features['R2'])
plt.ylabel('Training R2')
plt.xlabel('# features in model')
plt.savefig('data/figures/lasso r2 vs number of features.png')
plt.show()

"""
LASSO and OLS regression with statsmodels
"""
# pick favorite LASSO model
fav_lasso = lassolarsbic

# select alpha value
alpha = fav_lasso.alpha_

# standardize
X_train_sm = X_train.copy()
X_train_sm[X_train_sm.columns] = StandardScaler().fit_transform(X_train_sm)
y_train_sm = pd.DataFrame(y_train).copy()
y_train_sm[y_train_sm.columns] = StandardScaler().fit_transform(y_train_sm)

# add constants
X_train_sm = sm.add_constant(X_train)

# # fit model
# lasso_sm = sm.OLS(y_train_sm, X_train_sm)
# lasso_sm_res = lasso_sm.fit_regularized(alpha=alpha)
# print(lasso_sm_res.params)

## Linear regression with updated features - statsmodels
keep_features = [v for v,z in zip(X.columns, fav_lasso.coef_ != 0) if z]
ols_sm = sm.OLS(y_train_sm, X_train_sm[keep_features])
ols_sm_res = ols_sm.fit()
ols_sm_res.summary()

# top coefficients
print(f'Top positive coefs: \n{ols_sm_res.params.sort_values(ascending=False).head()}')
print(f'Top negative coefs: \n{ols_sm_res.params.sort_values(ascending=True).head()}')

# check normality in residuals
y_pred_sm = ols_sm_res.predict(X_train_sm[keep_features])

fig, axs = plt.subplots(1,2)
sns.histplot(ols_sm_res.resid, ax=axs[0])
sns.scatterplot(x=y_pred_sm, y=ols_sm_res.resid, ax=axs[1])
axs[1].set_ylabel('Residuals')
axs[1].set_xlabel('Predicted')
axs[0].set_xlabel('Residuals')
plt.savefig('data/figures/normality in residuals.png')
plt.show()
