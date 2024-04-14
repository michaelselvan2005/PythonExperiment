
import pandas as pd
dataset=pd.read_csv("50_Startups.csv")
dataset = pd.get_dummies(dataset, dtype=int,drop_first=True)
print(dataset.columns)
independent =dataset[['R&D Spend', 'Administration', 'Marketing Spend','State_Florida', 'State_New York']]
print(independent)
dependent=dataset[["Profit"]]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)



from sklearn.ensemble import RandomForestRegressor
regressor =RandomForestRegressor(n_estimators=50,random_state=0)
regressor.fit(X_train, y_train)


y_pred=regressor.predict(X_test)

from sklearn.metrics import r2_score
r_score=r2_score(y_test,y_pred)

print(r_score) # 0.944633639431341 Good Model

# To save the model by using Pickle

import pickle
filename="finalized_model_random_forest.sav"

pickle.dump(regressor,open(filename,'wb'))
loaded_model=pickle.load(open("finalized_model_random_forest.sav",'rb'))
result=loaded_model.predict([[130298.13,145530.06,323876.68,0,1]])

print("Result !!!!!",result)
