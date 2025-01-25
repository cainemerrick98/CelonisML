from unittest.mock import Mock, MagicMock
from CelonisML.core import CelonisML, DataExtractor, ModelTrainer
from CelonisML.utils import KPI
from pycelonis.pql import PQLColumn
import unittest
from pandas import DataFrame, Series
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score

class TestCelonisML(unittest.TestCase):

    def setUp(self):
        self.celonisml = CelonisML('data_model', 'knowledge_model')
        return super().setUp()
    
    def test_add_predictor_pql_column(self):
        self.celonisml.add_predictor(PQLColumn(name='attribute', query='"table"."attribute"'))
        self.assertEqual(1, len(self.celonisml.data_extractor.predictors))
    
    def test_add_predictor_knowledge_model_kpi(self):
        self.celonisml.add_predictor(KPI(id='kpi_id'))
        self.assertEqual(1, len(self.celonisml.data_extractor.predictors))

    def test_add_multiple_predictor_pql_column(self):
        self.celonisml.add_predictor(PQLColumn(name='attribute', query='"table"."attribute"'), PQLColumn(name='attribute', query='"table"."attribute"'))
        self.assertEqual(2, len(self.celonisml.data_extractor.predictors))
    
    def test_add_predictor_failure(self):
        with self.assertRaises(TypeError):
            self.celonisml.add_predictor('"table"."attribute"')

    def test_load_data_with_target(self):
        self.celonisml.data_extractor = Mock()
        self.celonisml.data_extractor.extract_data.return_value = DataFrame({'Col1':[1,2,3], 'Col2':[1,2,3]})
        self.celonisml.data_extractor.extract_pql_column.return_value = PQLColumn(name='Col1', query='query')

        self.celonisml.load_data()
        self.assertEqual(len(self.celonisml.data.columns), 2)
        self.assertEqual(len(self.celonisml.data), 3)
        self.assertEqual(self.celonisml.target_column, 'Col1')
    
    def test_add_model_correct(self):
        from sklearn.linear_model import LinearRegression
        self.celonisml.add_model(LinearRegression())
        self.assertIsNotNone(self.celonisml.model_trainer.model)
    
    def test_add_model_incorrect(self):
        with self.assertRaises(TypeError):
            mock = Mock()
            self.celonisml.add_model(mock)

class TestDataExtractor(unittest.TestCase):
    
    def setUp(self):
        return super().setUp()
    
    def test_set_saola_connector_dm_connector(self):
        data_extractor = DataExtractor()
        data_extractor.predictors = [PQLColumn(name='col1', query='col2')]
        data_extractor.target = PQLColumn(name='col2', query='col2')        
        self.assertFalse(data_extractor.requires_knowledge_model_connector())
    
    def test_set_saola_connector_km_connector(self):
        data_extractor = DataExtractor()
        data_extractor.predictors = [PQLColumn(name='col1', query='col2'), KPI('kpi_id')]
        data_extractor.target = PQLColumn(name='col2', query='col2')
        self.assertTrue(data_extractor.requires_knowledge_model_connector())

        data_extractor.predictors = [PQLColumn(name='col1', query='col2')]
        data_extractor.target = KPI(id='kpi_id')
        self.assertTrue(data_extractor.requires_knowledge_model_connector())

    def test_extract_pql_column_with_kpi(self):
        data_extractor = DataExtractor()
        mock_knowledge_model = MagicMock()
        mock_kpi = Mock(id='kpi_id', display_name='kpi', pql='pql_query')
        mock_knowledge_model.get_kpi.return_value = mock_kpi

        result = data_extractor.extract_pql_column(mock_knowledge_model, KPI('kpi_id'))

        self.assertIsInstance(result, PQLColumn)
        self.assertEqual(result.name, 'kpi')
        self.assertEqual(result.query, 'pql_query')
        
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