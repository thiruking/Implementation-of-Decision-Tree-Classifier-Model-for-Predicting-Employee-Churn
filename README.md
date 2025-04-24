# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data – Import the employee dataset with relevant features and churn labels.

2.Preprocess Data – Handle missing values, encode categorical features, and split into train/test sets.

3.Initialize Model – Create a DecisionTreeClassifier with desired parameters.

4.Train Model – Fit the model on the training data.

5.Evaluate Model – Predict on test data and check accuracy, precision, recall, etc.

6.Visualize & Interpret – Visualize the tree and identify key features influencing churn. 


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: THIRUMALAI K
RegisterNumber:  212224240176
*/
```
```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv(r"E:\Desktop\CSE\Introduction To Machine Learning\dataset\Employee.csv")
print(data.head())
print(data.info())
data.isnull().sum()
print(data["left"].value_counts())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
print(x.head())
y=data["left"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
pre=dt.predict([[0.5,0.8,9,260,6,0,1,2]])
print(pre)

plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/d10c8ad9-15ec-4f06-a35f-c003066147d4)
![image](https://github.com/user-attachments/assets/e8a401f8-1254-4b1a-b98a-131d7a59eeaf)
![image](https://github.com/user-attachments/assets/c195c757-2095-474a-9701-87687e164c73)
![image](https://github.com/user-attachments/assets/088f8038-27e7-415e-b095-6045f5dcb39c)
![image](https://github.com/user-attachments/assets/fe130b5b-5ef2-4efc-ba8e-97f3787c08ad)
![image](https://github.com/user-attachments/assets/564787e7-301f-4b74-8b6f-b0e502846216)
![image](https://github.com/user-attachments/assets/a5428fc7-f601-4725-a822-8f7c38a19594)
![image](https://github.com/user-attachments/assets/fca22ee7-6b52-44fd-be5f-34e1dbf0046b)










## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
