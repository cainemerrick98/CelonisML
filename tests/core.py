from unittest.mock import Mock, MagicMock
from CelonisML.core import CelonisML, DataExtractor
from CelonisML.utils import KPI
from pycelonis.ems.data_integration.data_model import DataModel
from pycelonis.ems.studio.content_node.knowledge_model import KnowledgeModel
from pycelonis.pql import PQL, PQLColumn
from pycelonis.pql.saola_connector import DataModelSaolaConnector, KnowledgeModelSaolaConnector
import unittest
from pandas import DataFrame

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
        

        