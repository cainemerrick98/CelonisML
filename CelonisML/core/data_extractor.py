from pycelonis.ems.data_integration.data_model import DataModel
from pycelonis.ems.studio.content_node.knowledge_model import KnowledgeModel
from pycelonis.pql import PQL, PQLColumn
from pycelonis.pql.data_frame import DataFrame as PQLDataFrame
from pycelonis.pql.saola_connector import DataModelSaolaConnector, KnowledgeModelSaolaConnector

def is_invalid_datafield(predictor):
    return not isinstance(predictor, (KPI, PQLColumn))


class DataExtractor():
    """
    Good explanation of the class
    """
    def __init__(self):
        self.predictors = []
        self.target = None
    
    def add_predictor(self, predictor):
        if is_invalid_datafield(predictor):
                raise TypeError('Predictor must be of type KPI or PQLColumn')
        self.predictors.append(predictor)
    
    def remove_predictor(self, predictor):
        if is_invalid_datafield(predictor):
                raise TypeError('Predictor must be of type KPI or PQLColumn')
        self.predictors.remove(predictor)
    
    def set_target(self, target):
        if is_invalid_datafield(target):
            raise TypeError('Target must be of type KPI or PQLColumn')
        self.target = target

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
        
        #TODO: maybe if knowledge model is passed then set KMSaola other wise DMSaola
        if self.requires_knowledge_model_connector():
            return PQLDataFrame.from_pql(query, 
            saola_connector=KnowledgeModelSaolaConnector(data_model=data_model, knowledge_model=knowledge_model)).to_pandas() 
        else:
            return PQLDataFrame.from_pql(query, 
            saola_connector=DataModelSaolaConnector(data_model=data_model)).to_pandas() 

    def extract_pql_column(self, knowledge_model, column):
        """
        ensures the column is a pql column before it is added to the query
        """
        if isinstance(column, PQLColumn):
            return column
        elif isinstance(column, KPI):
            kpi = knowledge_model.get_kpi(column.id_)
            return PQLColumn(name=kpi.display_name, query=kpi.pql)
        else:
            raise TypeError(f'column {column} is not of type PQLColumn or KPI')

    def requires_knowledge_model_connector(self):
        """
        checks if the `KnowledgeModelSaolaConnector` is required
        """
        columns = self.predictors + [self.target]
        if all(map(lambda x: isinstance(x, PQLColumn) or x is None , columns)):
            return False
        else:
            return True

class KPI():
    """
    represents a knowledge model KPI for users to add as predictors or targets
    """
    def __init__(self, id_:str):
        self.id_ = id_
    
    def __str__(self):
        return f"KPI(id='{self.id_}')"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, value):
        return isinstance(value, KPI) and value.id_ == self.id_
