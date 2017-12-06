---
title: Murder Prediction Models
notebook: Models.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}








```python
data = pd.read_csv('data_final.csv')
data = data.drop(['Unnamed: 0', 'Violent Crime', 'Rape', 'Robbery', 'Burglary',
                  'Aggravated Assault', 'Property Crime', 'Larceny-theft',
                  'Motor Vehicle Theft', 'State'], axis=1)
```




```python
X = data.drop(['Murder and nonnegligent manslaughter'], axis=1)
Y = data['Murder and nonnegligent manslaughter']
```




```python
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3)
```


## Baseline Model: Linear Regression



```python
linear = LinearRegression(fit_intercept=True).fit(xtrain, ytrain)

print('Linear Regression estimator\'s score on the training set is:',
     linear.score(xtrain, ytrain))

print('Linear Regression estimator\'s score on the test set is:',
     linear.score(xtest, ytest))
```


    Linear Regression estimator's score on the training set is: 0.783373293114
    Linear Regression estimator's score on the test set is: 0.705385591167


It appears that the Linear Regression does a decent job on both the training and test set

## Alternative Regression Approaches

### Lasso Regression



```python
lasso = LassoCV(fit_intercept=True, cv=5).fit(xtrain, ytrain)

print('Lasso Regression estimator\'s score on the training set is:',
     lasso.score(xtrain, ytrain))

print('Lasso Regression estimator\'s score on the test set is:',
     lasso.score(xtest, ytest))
```


    Lasso Regression estimator's score on the training set is: 0.0439594581424
    Lasso Regression estimator's score on the test set is: 0.0534207887922


Lasso regression performs really poorly on both test and training data and clearly underperforms Linear Regression

### Ridge Regression



```python
ridge = RidgeCV(fit_intercept=True, cv=5).fit(xtrain, ytrain)

print('Ridge Regression estimator\'s score on the training set is:',
     ridge.score(xtrain, ytrain))

print('Ridge Regression estimator\'s score on the test set is:',
     ridge.score(xtest, ytest))
```


    Ridge Regression estimator's score on the training set is: 0.765001838187
    Ridge Regression estimator's score on the test set is: 0.702764031499


It appears that Ridge regression significantly performs about as well as the Linear Regression

## Trees and Gradient Boosting

### Decision Tree Regressor



```python
tree = DecisionTreeRegressor()
depths = [i for i in range(2, 11)]
tree_param_gree = {'max_depth': depths}
tree_cv = GridSearchCV(tree, param_grid=tree_param_gree, n_jobs=-1, cv=5)
tree_cv.fit(xtrain, ytrain)

print('The cross validated max depth of the Decision Tree Regressor is:',
     tree_cv.best_params_.get('max_depth'))

print('Decision Tree Regressor\'s score on the training set:',
     tree_cv.score(xtrain, ytrain))

print('Decision Tree Regressor\'s score on the test set:',
     tree_cv.score(xtest, ytest))
```


    The cross validated max depth of the Decision Tree Regressor is: 7
    Decision Tree Regressor's score on the training set: 0.74326698995
    Decision Tree Regressor's score on the test set: 0.504114040828


It appears that the simple Decision Tree estimator performs worse than the Ridge and Linear Regressions

### Random Forest



```python
param_dict = OrderedDict(
    n_estimators = [200, 400, 600, 800],
    max_features = [0.2, 0.4, 0.6, 0.8])

results = {}
estimators= {}
for n, f in product(*param_dict.values()):
    params = (n, f)
    est = RandomForestRegressor(oob_score=True,
                                n_estimators=n, max_features=f, n_jobs=-1)
    est.fit(xtrain, ytrain)
    results[params] = est.oob_score_
    estimators[params] = est

outparams = max(results, key = results.get)

print('The optimal number of trees and max features for RF estimator are: %0.1f & %0.1f'
     %(outparams[0], outparams[1]))

rf = estimators[outparams]

print('RF score on the training set is:',
     rf.score(xtrain, ytrain))

print('RF score on the test set is:',
     rf.score(xtest, ytest))
```


    The optimal number of trees and max features for RF estimator are: 800.0 & 0.8
    RF score on the training set is: 0.95519341536
    RF score on the test set is: 0.718894539457


The Random Forest estimator clearly outperforms a single Decision Tree on both the test and training set, and does slightly better than the Linear and Ridge regressions

### Boosting



```python
gb = GradientBoostingRegressor(n_estimators=3000)


param_grid = {'learning_rate': [0.001, 0.01, 0.1],
              'max_features': [0.2, 0.4, 0.6, 0.8],
              'max_depth': [1, 3, 6],
              }
gb_cv = GridSearchCV(gb, param_grid, cv=5, n_jobs=-1)
gb_cv.fit(xtrain, ytrain)
```





    GridSearchCV(cv=5, error_score='raise',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_split=1e-07,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=3000,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'learning_rate': [0.001, 0.01, 0.1], 'max_features': [0.2, 0.4, 0.6, 0.8], 'max_depth': [1, 3, 6]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)





```python
print('The optimal learning rate, max depth, and max features for the Gradient Boosting estimator are: %f, %0.1f & %0.1f'
     %(gb_cv.best_params_.get('learning_rate'), gb_cv.best_params_.get('max_depth') ,gb_cv.best_params_.get('max_features')))

print('Gradient Boosting estimator\'s score on the training set is:',
     gb_cv.score(xtrain, ytrain))

print('Gradient Boosting estimator\'s score on the test set is:',
     gb_cv.score(xtest, ytest))
```


    The optimal learning rate, max depth, and max features for the Gradient Boosting estimator are: 0.010000, 6.0 & 0.4
    Gradient Boosting estimator's score on the training set is: 0.965179908418
    Gradient Boosting estimator's score on the test set is: 0.722487297994


It appears that the Gradient Boosting estimators slightly outperforms the Random Forest estimator on the training and test set

## Stacking



```python
models = [tree_cv, rf, gb_cv, linear, lasso, ridge]

predictions_train = np.zeros((xtrain.shape[0], len(models)))
predictions_test = np.zeros((xtest.shape[0], len(models)))


for i in range(len(models)):
    predictions_train[:,i] = models[i].predict(xtrain)
    predictions_test[:,i] = models[i].predict(xtest)
```


### Stacked Lasso



```python
stack_lasso = LassoCV(cv=5)
stack_lasso.fit(predictions_train, ytrain)

print('Stacked Lasso Regression estimator\'s score on the training set is:',
     stack_lasso.score(predictions_train, ytrain))

print('Stacked Lasso Regression estimator\'s score on the test set is:',
     stack_lasso.score(predictions_test, ytest))
```


    Stacked Lasso Regression estimator's score on the training set is: 0.986287494232
    Stacked Lasso Regression estimator's score on the test set is: 0.713117772029


### Stacked Ridge



```python
stack_ridge = RidgeCV(cv=5)

stack_ridge.fit(predictions_train, ytrain)

print('Stacked Ridge Regression estimator\'s score on the training set is:',
     stack_ridge.score(predictions_train, ytrain))

print('Stacked Ridge Regression estimator\'s score on the test set is:',
     stack_ridge.score(predictions_test, ytest))
```


    Stacked Ridge Regression estimator's score on the training set is: 0.986476884867
    Stacked Ridge Regression estimator's score on the test set is: 0.713748634317


### Stacked Random Forest



```python
s_param_dict = OrderedDict(
    n_estimators = [200, 400, 600, 800],
    max_features = [0.2, 0.4, 0.6, 0.8])

s_results = {}
s_estimators= {}
for n, f in product(*s_param_dict.values()):
    s_params = (n, f)
    s_est = RandomForestRegressor(oob_score=True,
                                n_estimators=n, max_features=f, n_jobs=-1)
    s_est.fit(predictions_train, ytrain)
    s_results[s_params] = s_est.oob_score_
    s_estimators[s_params] = s_est

s_outparams = max(s_results, key = s_results.get)

print('The optimal number of trees and max features for RF estimator are: %0.1f & %0.1f'
     %(s_outparams[0], s_outparams[1]))

stacked_rf = s_estimators[s_outparams]

print('Stacked RF score on the training set is:',
     stacked_rf.score(predictions_train, ytrain))

print('Stakced RF score on the test set is:',
     stacked_rf.score(predictions_test, ytest))
```


    The optimal number of trees and max features for RF estimator are: 400.0 & 0.8
    Stacked RF score on the training set is: 0.998214481645
    Stakced RF score on the test set is: 0.713514059896


### Stacked Boosting



```python
s_gb = GradientBoostingRegressor(n_estimators=3000)


s_param_grid = {'learning_rate': [0.001, 0.01, 0.1],
              'max_features': [0.2, 0.4, 0.6, 0.8],
              'max_depth': [1, 3, 6],
              }
s_gb_cv = GridSearchCV(s_gb, s_param_grid, cv=5, n_jobs=-1)
s_gb_cv.fit(predictions_train, ytrain)

print('The optimal learning rate, max depth, and max features for the Gradient Boosting estimator are: %f, %0.1f & %0.1f'
     %(s_gb_cv.best_params_.get('learning_rate'), s_gb_cv.best_params_.get('max_depth'),
       s_gb_cv.best_params_.get('max_features')))


print('Gradient Boosting estimator\'s score on the training set is:',
     s_gb_cv.score(predictions_train, ytrain))

print('Gradient Boosting estimator\'s score on the test set is:',
     s_gb_cv.score(predictions_test, ytest))
```


    The optimal learning rate, max depth, and max features for the Gradient Boosting estimator are: 0.010000, 3.0 & 0.8
    Gradient Boosting estimator's score on the training set is: 0.995637426977
    Gradient Boosting estimator's score on the test set is: 0.715357246931


It appears that Stacking our existing models improves prediction on the training set substantially but, does not improve upon the Randon Forest or Boosting estimators
