

import pandas as pd
dataset=pd.read_csv("insurance.csv")
dataset = pd.get_dummies(dataset, dtype=int,drop_first=True)
print(dataset.columns)

# Index(['age', 'bmi', 'children', 'expenses', 'sex_male', 'smoker_yes',
       #'region_northwest', 'region_southeast', 'region_southwest'],
      #dtype='object')

      #https://www.kaggle.com/datasets/awaiskaggler/insurance-csv


independent =dataset[['age', 'bmi', 'children', 'expenses', 'sex_male', 'smoker_yes','region_northwest', 'region_southeast', 'region_southwest']]
print(independent)
dependent=dataset[["bmi"]]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)

# Here, we have to import DecisionTreeRegressor
# Creating Model

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(criterion='poisson',splitter='random')
regressor.fit(X_train,y_train)

import matplotlib.pyplot as plt
from sklearn import tree
tree.plot_tree(regressor)
plt.show()
# weight=regressor.coef_
#weight


# bais=regressor.intercept_
# bais
# By using the predict() we can predict by using TEST data.

# Evaluation
y_pred=regressor.predict(X_test)

from sklearn.metrics  import r2_score
r_score=r2_score(y_test,y_pred)

print(r_score)
