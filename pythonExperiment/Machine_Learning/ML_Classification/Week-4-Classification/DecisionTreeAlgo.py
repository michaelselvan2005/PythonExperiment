import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Social_Network_Ads.csv")

print(dataset)

dataset=pd.get_dummies(dataset,drop_first=True)

print(dataset)


dataset=dataset.drop("User ID",axis=1)

print(dataset)

dataset["Purchased"].value_counts()

indep=dataset[["Age","EstimatedSalary","Gender_Male"]]
dep=dataset["Purchased"]

indep.shape

dep

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(indep, dep, test_size = 1/3, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

classifier.predict([[40,300,1]])

from sklearn.metrics import classification_report
clf_report = classification_report(y_test, y_pred)

print(clf_report)

age_input=float(input("Age:"))
salary_input=float(input("BMI:"))
sex_male_input=int(input("Sex Male 0 or 1:"))

classifier.predict([[age_input,salary_input,sex_male_input]])

dir(clf_report)

