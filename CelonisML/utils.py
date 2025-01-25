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
    