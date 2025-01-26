from CelonisML.core.data_extractor import DataExtractor, KPI
import unittest
from unittest.mock import Mock, MagicMock
from pycelonis.pql import PQLColumn
class TestDataExtractor(unittest.TestCase):
    
    def setUp(self):
        return super().setUp()
    
    def test_requires_saola_connector_dm_connector(self):
        data_extractor = DataExtractor()
        data_extractor.predictors = [PQLColumn(name='col1', query='col2')]
        data_extractor.target = PQLColumn(name='col2', query='col2')        
        self.assertFalse(data_extractor.requires_knowledge_model_connector())
    
    def test_requires_saola_connector_dm_connector_target_is_none(self):
        data_extractor = DataExtractor()
        data_extractor.predictors = [PQLColumn(name='col1', query='col2')]
        data_extractor.target = None
        self.assertFalse(data_extractor.requires_knowledge_model_connector())
    
    def test_requires_saola_connector_km_connector(self):
        data_extractor = DataExtractor()
        data_extractor.predictors = [PQLColumn(name='col1', query='col2'), KPI('kpi_id')]
        data_extractor.target = PQLColumn(name='col2', query='col2')
        self.assertTrue(data_extractor.requires_knowledge_model_connector())

        data_extractor.predictors = [PQLColumn(name='col1', query='col2')]
        data_extractor.target = KPI(id_='kpi_id')
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
        