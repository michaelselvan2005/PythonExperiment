import pandas as pd

dataset=pd.read_csv("50_Startups.csv")

print(dataset)

dataset=pd.get_dummies(dataset,dtype=int,drop_first=True)

print(dataset)

dataset.columns
independent =dataset[['R&D Spend', 'Administration', 'Marketing Spend','State_Florida', 'State_New York']]

print(independent)
 

dependent=dataset[["Profit"]]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)

print("X-Train",X_train)
print("X-Test",X_test)
print("Y-Train",y_train)
print("Y-Test",y_test)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.svm import SVR
regressor=SVR(kernel="rbf",C=1000000000) # SVM -> Non-Linear using one of the algorithm
                                # here, we are saying Linear/Non-Linear we use mention by using Kernel
regressor.fit(X_train,y_train)

print(regressor)

y_pred=regressor.predict(X_test)

from sklearn.metrics  import r2_score
r_score=r2_score(y_test,y_pred)

print(r_score)

