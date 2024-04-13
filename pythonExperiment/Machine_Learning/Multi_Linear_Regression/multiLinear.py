
import pandas as pd
dataset=pd.read_csv("50_Startups.csv")

#dataset 

# Now we have to perform pre-processing
# AFTER SEEING CATEGORICAL DATA FIRST THING COME INTO MINDE IS Nominal /Orinal
# NOMIMAL MEANS One Hot Encoding (Colimn expansation will heppen)

#dataset=pd.get_dummies(dataset)
# Initially it was showing all the columns, not it will show 1,0,0 means one columns 
#sepearted as three columns (Column Expansaion)
# One disavvange is reperation of columns are coming we have to drop any of the dummy columns


#dataset


#dataset=pd.get_dummies(dataset,drop_first=True)  ?? WHY DUMMIES 
dataset = pd.get_dummies(dataset, dtype=int,drop_first=True)




#print(dataset)

# Split Input & Output

print(dataset.columns)

#Index(['R&D Spend', 'Administration', 'Marketing Spend', 'Profit','State_Florida', 'State_New York'],
 #     dtype='object')



independent =dataset[['R&D Spend', 'Administration', 'Marketing Spend','State_Florida', 'State_New York']]

print(independent)
 

dependent=dataset[["Profit"]]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)

print("X-Train",X_train)
print("X-Test",X_test)
print("Y-Train",y_train)
print("Y-Test",y_test)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


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


# To save the model by using Pickle

import pickle
filename="finalized_model_multi_linear.sav"

pickle.dump(regressor,open(filename,'wb'))
loaded_model=pickle.load(open("finalized_model_multi_linear.sav",'rb'))
result=loaded_model.predict([[130298.13,145530.06,323876.68,0,1]])

print("Result !!!!!",result)
