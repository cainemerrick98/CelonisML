class KPI():
    """
    represents a knowledge model KPI.
    for users to add as predictors or targets
    """
    def __init__(self, id:str):
        self.id = id
    
    def __str__(self):
        return f"KPI(id='{self.id}')"

    def __repr__(self):
        return self.__str__()