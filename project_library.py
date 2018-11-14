import numpy as np
import pandas as pd

#functions to write and open data in the HDF5 format with metadata
#source https://stackoverflow.com/questions/29129095/save-additional-attributes-in-pandas-dataframe/29130146#29130146
def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(filename):
    with pd.HDFStore(filename) as store:
        data = store['mydata']
        metadata = store.get_storer('mydata').attrs.metadata
        return data, metadata
