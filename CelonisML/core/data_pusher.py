from pycelonis.ems.data_integration.data_pool import DataPool
from pandas import DataFrame
class DataPusher():
    
    def __init__():
        pass

    def upload_data(self, data:DataFrame, data_pool:DataPool, name:str):
        """
        simply adds the dataframe to the data pool
        """
        data_pool.create_table(
            df=data,
            table_name=name,
            drop_if_exists=True
        )