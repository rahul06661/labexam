import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

dataSet=pd.read_csv(r"C:\Users\SJCET\Downloads\diabetes.csv")
print(dataSet.columns)
x=dataSet[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']].values
y=dataSet['Outcome'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model=KNeighborsClassifier(n_neighbors=4)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy_score",accuracy_score(y_pred,y_test))
print("classification report ",classification_report(y_pred,y_test))




