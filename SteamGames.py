import pandas as pd #This module is used to read the csv files and store in the dataframe
import numpy as np  #This module is used for numerical column checking and counting
import seaborn as sns   #This module is used to visualize the data provided
import matplotlib.pyplot as plt #This module prints out the visualized data
from sklearn.preprocessing import StandardScaler     #The StandardScaler module is used to scale data needed for testing and training
from sklearn.preprocessing import LabelEncoder  #This module is used to convert categorical values into numerical columns for preprocessing
from sklearn.model_selection import train_test_split    #This module is used for data testing
from sklearn.linear_model import LogisticRegression  #This module is used to predict discrete categorical values
from sklearn.ensemble import RandomForestClassifier #This module is used for learning methods for classification, regression and it operates by constructing a multitude of decision trees at training time 
from sklearn.svm import SVC #It contains SVM(Support Vector Machine) and finds the hyperplane that best separates the data points into different classes.
from sklearn.metrics import accuracy_score #This module checks for accuracy of imported algorithms

data=pd.read_csv("C:/Users/kaust/OneDrive/Desktop/AIML ADG/SteamGames/steam-games.csv")

#Data Description
print(data.head())  #Prints out first five row values
print(data.tail())  #Prints out last five row values
print(data.shape)  #Prints out no of rows and columns
print(data.describe())  #Prints out the statistical data such as count,mean etc for data have integer values
print(data.info())  #Prints out the the column heads, the no of values it has stored along with the data type present
print(data.nunique())  #Prints out the amount of data that is unique in each column

#Data Cleaning
print(data.isnull().sum(),"\n")  #Prints out missing values of data
numeric_columns = data.select_dtypes(include=[np.number]).columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())  # Fill numeric columns with their mean
print(data.duplicated().sum(),"\n")  #Prints out duplicated values
categorical_columns = data.select_dtypes(include=[object]).columns
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col]) # Encodes the categorical variables by using LabelEncoder

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

#Data Testing and preprocessing
X=data.drop('recent_review_count',axis=1) #This is the feature matrix and it contains all the independent variables needed for model training, excluding the target variable.
y=data['recent_review_count']  #This is the target variable which will be used as dependent variable for the machine learning model.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
sc_X=StandardScaler()   #Used to standardize data by making mean=0 and standard deviation=1
X_train=sc_X.fit_transform(X_train) #Method used to do the standardization
X_test=sc_X.transform(X_test)  #Secondary method
print(X_train,"\n") #Used to print out the standardized data
print(X_test)

#Model Training using Machine Learning Algorithms
#1 Logistic Regression

logR=LogisticRegression(random_state=0)
logR.fit(X_train,y_train) #This is used to fit logistic results into test dataset
y_pred_logR=logR.predict(X_test)
accuracy_logR=accuracy_score(y_test, y_pred_logR) #Accuracy check for Logistic Regression

#2 Random Forest
rf_clf=RandomForestClassifier(random_state=0)
rf_clf.fit(X_train,y_train)    #This is used to fit randomforest results into test dataset
y_pred_rf=rf_clf.predict(X_test)
accuracy_rf=accuracy_score(y_test, y_pred_rf) #Accuracy check for random forest

#3 Support Vector Machine
svm_clf=SVC(random_state=0)
svm_clf.fit(X_train, y_train)   #This is used to fit SVM results into test dataset
y_pred_svm=svm_clf.predict(X_test)
accuracy_svm=accuracy_score(y_test, y_pred_svm)   #Accuracy check for SVM

print("Accuracy of Logistic Regression:", accuracy_logR)
print("Accuracy of Random Forest:", accuracy_rf)
print("Accuracy of Support Vector Machine:", accuracy_svm)
