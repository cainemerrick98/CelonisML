from pycelonis.ems.data_integration.data_model import DataModel
from pycelonis.ems.studio.content_node.knowledge_model import KnowledgeModel
from .data_extractor import DataExtractor
from .model_trainer import ModelTrainer

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
            self.data_extractor.add_predictor(predictor)

    def remove_predictor(self, *predictors):
        """
        adds one or more predictors to the model
        """
        for predictor in predictors:
            self.data_extractor.remove_predictor(predictor)

    def set_target(self, target):
        """
        sets the target variable to the model for supervised
        learning tasks.
        """
        self.data_extractor.set_target(target)
    
    def load_data(self):
        """
        runs the pql query to extract the data for current dataset of predictors and target
        """
        if self.data_extractor.target:
            self.target_column = self.data_extractor.extract_pql_column(self.knowledge_model, self.data_extractor.target).name

        self.data = self.data_extractor.extract_data(self.data_model, self.knowledge_model)

    def add_model(self, model):
        self.model_trainer.set_model(model)

    def train_model(self):
        """
        trains the model and 
        """
        if self.data is None:
            raise ValueError('Data is `None`. Ensure you have extracted data from Celonis before training the model')
        
        if self.target_column is None:
            return self.model_trainer.train_and_evaluate(self.data)

        preditors = self.data.drop(columns=[self.target_column])
        target = self.data[self.target_column]

        return self.model_trainer.train_and_evaluate(preditors, target)
    