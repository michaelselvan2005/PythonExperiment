import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset=pd.read_csv("insurance.csv")
dataset = pd.get_dummies(dataset, dtype=int,drop_first=True)
print(dataset.columns)

# Index(['age', 'bmi', 'children', 'expenses', 'sex_male', 'smoker_yes',
       #'region_northwest', 'region_southeast', 'region_southwest'],
      #dtype='object')

      #https://www.kaggle.com/datasets/awaiskaggler/insurance-csv


independent =dataset[['age', 'bmi', 'children','sex_male','smoker_yes']]
print(independent)
dependent=dataset[["expenses"]]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,dependent,test_size=1/3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
param_grid ={'criterion':['mse','mae','friedman_mse'],
             'max_features':['auto','sqrt','log2'],
             'splitter':['best','random']}

grid=GridSearchCV(DecisionTreeRegressor(),param_grid,refit=True,verbose=3,n_jobs=-1)
grid.fit(X_train,y_train)

re=grid.cv_results_
print("The R_score value for best parameter {}:".format(grid.best_params_))

table=pd.DataFrame.from_dict(re)
print(table)

age_input=float(input("Age"))
bmi_input=float(input("BMI"))
children_input=float(input("Children"))
sex_male_input=int(input("Sex Male 0 or 1"))
smoker_yes_input=int(input("Smoker yes 0 or 1:"))



Future_Predition=grid.predict([[age_input,bmi_input,children_input,sex_male_input,smoker_yes_input]])
print("Future_Predition={}".format(Future_Predition))
