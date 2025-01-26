from pycelonis.pql import PQL, PQLColumn
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from pandas import DataFrame, Series

class ModelTrainer():
    """
    Good explanation of the class
    """
    def __init__(self):
        self.model = None
        self.scoring_func = None

    def set_model(self, model):
        if not isinstance(model, BaseEstimator):
            raise TypeError('Model must be an instance of BaseEstimatior')

        for method in ['fit', 'predict']:
            if not hasattr(model, method):
                raise TypeError('Model must be implement a `fit` and `predict` method')

        if not hasattr(model, '_estimator_type'):
            raise TypeError('Model must have an `_estimator_type` attribute')
        
        self.model = model
        if self.model._estimator_type == 'regressor':
            self.scoring_func = mean_squared_error
        elif self.model._estimator_type == 'classifier':
            self.scoring_func = accuracy_score
        else:
            self.scoring_func = silhouette_score

    def train_and_evaluate(self, X:DataFrame, y:Series=None):
        if not self.model:
            raise ValueError('No model set. Call `.set_model(model) first')
        
        if self.model._estimator_type in ['regressor', 'classifier']:
            if y is None:
                raise ValueError(f'{self.model} requires a target to be added')    
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            self.model.fit(X_train, y_train)
            return self.scoring_func(y_test, self.model.predict(X_test))
        else:
            X_train, X_test = train_test_split(X)
            self.model.fit(X_train)
            return self.scoring_func(X_test, self.model.predict(X_test))

    def predict(self, predictors:DataFrame)->Series:
        #TODO: assert model has been fit
        return self.model.predict(predictors)
