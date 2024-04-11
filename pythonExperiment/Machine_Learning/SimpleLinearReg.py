import pandas as pd

dataset=pd.read_csv("D:\i-n-MLOps\Project\PythonExperiment\PythonExperiment\pythonExperiment\Machine_Learning\Simple_Linear_Regression\Salary_Data.csv")

independent=dataset[["YearsExperience"]]

dependent=dataset[["Salary"]]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)

print("X-Train",X_train)
print("X-Test",X_test)
print("Y-Train",y_train)
print("Y-Test",y_test)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

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