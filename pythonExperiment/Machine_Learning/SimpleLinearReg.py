


# It will import all required functions to create a table
import pandas as pd


# It will do the data reading from the file

dataset=pd.read_csv("D:\i-n-MLOps\Project\PythonExperiment\PythonExperiment\pythonExperiment\Machine_Learning\Simple_Linear_Regression\Salary_Data.csv")

# Assigning input to the Variable
independent=dataset[["YearsExperience"]]

# Assigning output to the Variable
dependent=dataset[["Salary"]]


# Going to split the Training & Testing Data 70% for training data an 30% for testing data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)

print("X-Train",X_train)
print("X-Test",X_test)
print("Y-Train",y_train)
print("Y-Test",y_test)

# Applying the ALGORITHM to the given dataset , once we apply the fix() and passing the traing data
# Then it will create the Model for us.

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# By using the predict() we can predict by using TEST data.
y_pred=regressor.predict(X_test)


# By using the R2 we can find out the accuracy of the MODEL

from sklearn.metrics  import r2_score
r_score=r2_score(y_test,y_pred)

print(r_score)


# To save the model by using Pickle

import pickle
filename="finalized_model_linear.sav"

pickle.dump(regressor,open(filename,'wb'))
loaded_model=pickle.load(open("finalized_model_linear.sav",'rb'))
result=loaded_model.predict([[int(input("Enter the number of Years"))]])

print("Result !!!!!",result)

# Input is 0 that is Bias (Origin)
#Enter the number of Years0
#C:\Users\micha\anaconda3-new\envs\pythonexp\lib\site-packages\sklearn\base.py:465: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
#  warnings.warn(
#Result !!!!! [[26777.3913412]]