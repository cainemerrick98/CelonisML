"""
Core data classes are data extraction, model trainer and data upload
"""
from pycelonis.ems.data_integration.data_model import DataModel
from pycelonis.ems.studio.content_node.knowledge_model import KnowledgeModel
# from pycelonis.ems.studio.content_node.knowledge_model.kpi import Kpi
from pycelonis.pql import PQL, PQLColumn
from pycelonis.pql.data_frame import DataFrame as PQLDataFrame
from pycelonis.pql.saola_connector import DataModelSaolaConnector, KnowledgeModelSaolaConnector, verify_columns
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from utils import KPI
from pandas import DataFrame, Series

class CelonisML():
    """
    Good explanation of the class
    """
    def __init__(self, data_model:DataModel, knowledge_model:KnowledgeModel):
        self.data_extractor = DataExtractor()
        self.model_trainer = ModelTrainer()
        self.data_model = data_model
        self.knowledge_model = knowledge_model
        self.data = None

    def add_predictor(self, predictor):
        """
        adds a predictor variable to the model
        """
        #TODO unpack predictors
        if not isinstance(predictor, (KPI, PQLColumn)):
            raise TypeError('Predictor must be a knowledge model KPI or a PQL column')
        self.data_extractor.predictors.append(predictor)

    def add_target(self, target):
        """
        adds the target variable to the model for supervised
        learning tasks.
        """
        if not isinstance(target, (KPI, PQLColumn)):
            raise TypeError('Target must be a knowledge model KPI or a PQL column')
        self.data_extractor.target = target
    
    def load_data(self):
        """
        runs the pql query to extract the data for current set of predictors and target
        """
        self.data_extractor.set_saola_connector()
        self.data = self.data_extractor.extract_data(self.data_model, self.knowledge_model)

    def set_model(self, model:BaseEstimator):
        self.model_trainer.model = model

class DataExtractor():
    """
    Good explanation of the class
    """
    def __init__(self):
        self.predictors = []
        self.target = None
        self.saola_connector = None

    def extract_data(self, data_model:DataModel, knowledge_model:KnowledgeModel):
        """
        builds and runs the PQL query to extract the data from celonis. A pandas dataframe
        is returned
        """
        query = PQL()
        for predictor in self.predictors:
            query += self.extract_pql_column(knowledge_model, predictor)
        
        if self.target:
            query += self.extract_pql_column(knowledge_model, self.target)
        
        return PQLDataFrame.from_pql(query, saolaconnector=self.saola_connector).to_pandas() 

    def extract_pql_column(self, knowledge_model, column):
        """
        ensures the column is a pql column before it is added to the query
        """
        if isinstance(column, PQLColumn):
            return column
        elif isinstance(column, KPI):
            kpi = knowledge_model.get_kpi(column.id)
            return PQLColumn(name=kpi.display_name, query=kpi.pql)
        else:
            raise TypeError('predictor should be of type PQLColumn or KPI')#TODO:set a limit here?
    
    def set_saola_connector(self):
        """
        sets the correct saoloa connector given the set of predictors and the target
        """
        columns = self.predictors + [self.target]
        if all(map(lambda x: isinstance(x, PQLColumn), columns)):
            return DataModelSaolaConnector
        else:
            return KnowledgeModelSaolaConnector

class ModelTrainer():
    """
    Good explanation of the class
    """
    def __init__(self):
        self.model = None
    
    def train_model(self, target:Series, predictors:DataFrame):
        if not self.model:
            raise ValueError('No model set. Call `.set_model(model) first')
        
        X_train, X_test, y_train, y_test = train_test_split(predictors, target)

        #TODO: error testing for fit method
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self):
        ...

    def predict(self, predictors:DataFrame)->Series:
        return self.model.predict(predictors)

class DataPusher():
    ...