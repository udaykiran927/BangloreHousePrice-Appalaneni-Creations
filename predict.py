import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df=pd.read_csv("BangloreHouseData.csv")
df["location"]=df["location"].apply(lambda x:x.strip())
regg=LinearRegression()

pickle.dump(regg,open("model.pkl","wb"))