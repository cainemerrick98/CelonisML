from CelonisML.core.celonis_ml import CelonisML
from pycelonis.pql import PQLColumn
from CelonisML.core.data_extractor import KPI
import unittest
from unittest.mock import Mock
from pandas import DataFrame
import numpy as np

class TestCelonisML(unittest.TestCase):

    def setUp(self):
        self.celonisml = CelonisML('data_model', 'knowledge_model')
        return super().setUp()
    
    def test_add_predictor_pql_column(self):
        self.celonisml.add_predictor(PQLColumn(name='attribute', query='"table"."attribute"'))
        self.assertEqual(1, len(self.celonisml.data_extractor.predictors))
    
    def test_add_predictor_knowledge_model_kpi(self):
        self.celonisml.add_predictor(KPI(id_='kpi_id'))
        self.assertEqual(1, len(self.celonisml.data_extractor.predictors))

    def test_add_multiple_predictor_pql_column(self):
        self.celonisml.add_predictor(PQLColumn(name='attribute', query='"table"."attribute"'), PQLColumn(name='attribute', query='"table"."attribute"'))
        self.assertEqual(2, len(self.celonisml.data_extractor.predictors))
    
    def test_add_predictor_failure(self):
        with self.assertRaises(TypeError):
            self.celonisml.add_predictor('"table"."attribute"')

    def test_remove_predictor_pql_column(self):
        self.celonisml.add_predictor(PQLColumn(name='attribute', query='"table"."attribute"'))
        self.celonisml.remove_predictor(PQLColumn(name='attribute', query='"table"."attribute"'))
        self.assertEqual(0, len(self.celonisml.data_extractor.predictors))
    
    def test_remove_predictor_knowledge_model_kpi(self):
        self.celonisml.add_predictor(KPI(id_='kpi_id'))
        self.celonisml.remove_predictor(KPI(id_='kpi_id'))
        self.assertEqual(0, len(self.celonisml.data_extractor.predictors))
    
    def test_remove_predictor_knowledge_model_kpi_wrong_id(self):
        self.celonisml.add_predictor(KPI(id_='kpi_id'))
        with self.assertRaises(ValueError):
            self.celonisml.remove_predictor(KPI(id_='kpi_i'))
        

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

    def test_train_model(self):
        from sklearn.linear_model import LinearRegression
        self.celonisml.data = DataFrame({
            'Col1':np.random.normal(0, 1, 100),
            'Col2':np.random.normal(-2, 2, 100)
        })
        self.celonisml.data['Target'] = 3 * self.celonisml.data['Col1'] + 2 * self.celonisml.data['Col2'] + np.random.normal(0, 1, 100)
        self.celonisml.target_column = 'Target'
        self.celonisml.add_model(LinearRegression())

        score = self.celonisml.train_model()
        self.assertIsInstance(score, float)
