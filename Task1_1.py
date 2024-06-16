import pandas as pd
import numpy as np
import seaborn as sns

data=pd.read_csv("C:/Users/kaust/OneDrive/Desktop/AIML ADG/steam-games.csv")
print(data.head())  #Prints out first five row values
print(data.tail())  #Prints out last five row values
print(data.shape)  #Prints out no of rows and columns
print(data.describe())  #Prints out the statistical data such as count,mean etc for data have integer values
print(data.info())  #Prints out the the column heads, the no of values it has stored along with the data type present
print(data.nunique())  #Prints out the amount of data that is unique in each column
