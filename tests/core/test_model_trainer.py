from CelonisML.core.model_trainer import ModelTrainer
from pandas import DataFrame, Series
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
import unittest

class TestModelTrainer(unittest.TestCase):
    
    def setUp(self):
        self.model_trainer = ModelTrainer()
        self.X = DataFrame({
            'Col1':np.random.normal(0, 1, 100),
            'Col2':np.random.normal(-2, 2, 100)
        })
        self.y_reg = Series(5 + 2 * self.X['Col1'] + 0.5 * self.X['Col2'] + np.random.normal(0, 1, 100)) #regression equation
        self.y_clf = self.y_reg >= self.y_reg.mean()
        return super().setUp()
    
    def test_set_model_regressor(self):
        self.model_trainer.set_model(LinearRegression())
        self.assertEqual(self.model_trainer.model._estimator_type, 'regressor')
        self.assertIs(self.model_trainer.scoring_func, mean_squared_error)
    
    def test_train_and_evaluate_regression(self):
        self.model_trainer.set_model(LinearRegression())
        score = self.model_trainer.train_and_evaluate(self.X, self.y_reg)
        self.assertIsInstance(score, float)
    
    def test_set_model_classifier(self):
        self.model_trainer.set_model(LogisticRegression())
        self.assertEqual(self.model_trainer.model._estimator_type, 'classifier')
        self.assertIs(self.model_trainer.scoring_func, accuracy_score)

    def test_train_and_evaluate_classifier(self):
        self.model_trainer.set_model(LogisticRegression())
        score = self.model_trainer.train_and_evaluate(self.X, self.y_clf)
        self.assertIsInstance(score, float)
    
    def test_set_model_unsupervised(self):
        self.model_trainer.set_model(KMeans())
        self.assertEqual(self.model_trainer.model._estimator_type, 'clusterer')
        self.assertIs(self.model_trainer.scoring_func, silhouette_score)
    
    def test_train_and_evaluate_clusterer(self):
        self.model_trainer.set_model(KMeans())
        score = self.model_trainer.train_and_evaluate(self.X)
        self.assertIsInstance(score, float)
        
    def test_predict(self):
        self.model_trainer.set_model(LinearRegression())
        score = self.model_trainer.train_and_evaluate(self.X, self.y_reg)
        predictions = self.model_trainer.predict(self.X)
        self.assertIsInstance(predictions, np.ndarray)