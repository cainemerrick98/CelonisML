from unittest.mock import Mock, MagicMock
from CelonisML.core import CelonisML, DataExtractor, KPI
from pycelonis.ems.data_integration.data_model import DataModel
from pycelonis.ems.studio.content_node.knowledge_model import KnowledgeModel
from pycelonis.pql import PQL, PQLColumn
from pycelonis.pql.saola_connector import DataModelSaolaConnector, KnowledgeModelSaolaConnector

import unittest

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
    
    def test_add_predictor_failure(self):
        with self.assertRaises(TypeError):
            self.celonisml.add_predictor('"table"."attribute"')
    

class TestDataExtractor(unittest.TestCase):
    
    def setUp(self):
        return super().setUp()
    
    def test_set_saola_connector_dm_connector(self):
        data_extractor = DataExtractor()
        data_extractor.predictors = [PQLColumn(name='col1', query='col2')]
        data_extractor.target = PQLColumn(name='col2', query='col2')

        saola_connector = data_extractor.set_saola_connector()
        
        self.assertIs(saola_connector, DataModelSaolaConnector)
    
    def test_set_saola_connector_km_connector(self):
        data_extractor = DataExtractor()
        data_extractor.predictors = [PQLColumn(name='col1', query='col2'), KPI('kpi_id')]
        data_extractor.target = PQLColumn(name='col2', query='col2')

        saola_connector = data_extractor.set_saola_connector()
    
        self.assertIs(saola_connector, KnowledgeModelSaolaConnector)

        data_extractor.predictors = [PQLColumn(name='col1', query='col2')]
        data_extractor.target = KPI(id='kpi_id')

        saola_connector = data_extractor.set_saola_connector()


    
    def test_extract_pql_column_with_kpi(self):
        data_extractor = DataExtractor()
        mock_knowledge_model = MagicMock()
        mock_kpi = Mock(id='kpi_id', display_name='kpi', pql='pql_query')
        mock_knowledge_model.get_kpi.return_value = mock_kpi

        result = data_extractor.extract_pql_column(mock_knowledge_model, KPI('kpi_id'))

        self.assertIsInstance(result, PQLColumn)
        self.assertEqual(result.name, 'kpi')
        self.assertEqual(result.query, 'pql_query')
        

        