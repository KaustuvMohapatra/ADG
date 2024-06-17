import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/kaust/OneDrive/Desktop/AIML ADG/steam-games.csv")

#Data Description
print(data.head())  #Prints out first five row values
print(data.tail())  #Prints out last five row values
print(data.shape)  #Prints out no of rows and columns
print(data.describe())  #Prints out the statistical data such as count,mean etc for data have integer values
print(data.info())  #Prints out the the column heads, the no of values it has stored along with the data type present
print(data.nunique())  #Prints out the amount of data that is unique in each column

#Data Cleaning
print(data.isnull().sum())  #Prints out missing values of data
numeric_columns = data.select_dtypes(include=[np.number]).columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())  # Fill numeric columns with their mean
print(data.duplicated().sum())  #Prints out duplicated values

#Data Visualization
data.hist(bins=30, figsize=(10, 10))
plt.show()  #Prints out histogram of data
for column in numeric_columns:
    sns.kdeplot(data[column], shade=True)
    plt.title(f'Distribution of {column}')
    plt.show()  #Prints out KDE Plots
numeric_data = data[numeric_columns]
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix,xticklabels=correlation_matrix.columns,yticklabels=correlation_matrix.columns,annot=True)
plt.title('Correlation Matrix')
plt.show()  #Prints out a heatmap by using correlation matrix
