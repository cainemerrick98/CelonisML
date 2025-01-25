"""
Core data classes are data extraction, model trainer and data upload
"""
from pycelonis.ems.data_integration.data_model import DataModel
from pycelonis.ems.studio.content_node.knowledge_model import KnowledgeModel
from pycelonis.pql import PQL, PQLColumn
from pycelonis.pql.data_frame import DataFrame as PQLDataFrame
from pycelonis.pql.saola_connector import DataModelSaolaConnector, KnowledgeModelSaolaConnector
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from .utils import KPI
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
        self.target_column = None

    def add_predictor(self, *predictors):
        """
        adds one or more predictors to the model
        """
        for predictor in predictors:
            if not isinstance(predictor, (KPI, PQLColumn)):
                raise TypeError('Predictor must be of type KPI or PQLColumn')
            self.data_extractor.predictors.append(predictor)

    def add_target(self, target):
        """
        adds the target variable to the model for supervised
        learning tasks.
        """
        if not isinstance(target, (KPI, PQLColumn)):
            raise TypeError('Target must be of type KPI or PQLColumn')
        self.data_extractor.target = target
    
    def load_data(self):
        """
        runs the pql query to extract the data for current dataset of predictors and target
        """
        if self.data_extractor.target:
            self.target_column = self.data_extractor.extract_pql_column(self.knowledge_model, self.data_extractor.target).name

        self.data = self.data_extractor.extract_data(self.data_model, self.knowledge_model)

    def add_model(self, model:BaseEstimator):
        if not isinstance(model, BaseEstimator):
            raise TypeError('Model must be an instance of BaseEstimatior')

        for method in ['fit', 'predict']:
            if not hasattr(model, method):
                raise TypeError('Model must be implement a `fit` and `predict` method')

        if not hasattr(model, '_estimator_type'):
            raise TypeError('Model must have an `_estimator_type` attribute')


        self.model_trainer.set_model(model)

    def train_model(self):
        """
        trains the model and 
        """
        if self.data is None:
            raise ValueError('Data is `None`. Ensure you have extracted data from Celonis before training the model')
        
        if self.target_column is None:
            score = self.model_trainer.train_and_evaluate(self.data)

        target = self.data[self.target_column]
        preditors = self.data.drop(columns=[self.target_column])
        


class DataExtractor():
    """
    Good explanation of the class
    """
    def __init__(self):
        self.predictors = []
        self.target = None

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
        
        if self.requires_knowledge_model_connector():
            return PQLDataFrame.from_pql(query, 
            saolaconnector=KnowledgeModelSaolaConnector(data_model=data_model, knowledge_model=knowledge_model)).to_pandas() 
        else:
            return PQLDataFrame.from_pql(query, 
            saolaconnector=DataModelSaolaConnector(data_model=data_model)).to_pandas() 

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
            raise TypeError(f'column {column} is not of type PQLColumn or KPI')

    def requires_knowledge_model_connector(self):
        """
        checks if the `KnowledgeModelSaolaConnector` is required
        """
        columns = self.predictors + [self.target]
        if all(map(lambda x: isinstance(x, PQLColumn), columns)):
            return False
        else:
            return True

class ModelTrainer():
    """
    Good explanation of the class
    """
    def __init__(self):
        self.model = None
        self.scoring_func = None

    def set_model(self, model):
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

class DataPusher():
    ...