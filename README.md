# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset and required libraries.
2. Separate input (Hours) and output (Scores).
3. Split data into training and testing sets.
4. Train the Linear Regression model.
5. Predict results and evaluate performance.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VEDARANJAN S
RegisterNumber: 212225220119

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv(r"C:\Users\acer\Downloads\student_scores.csv")
df.head(10)

```

<img width="1072" height="552" alt="image" src="https://github.com/user-attachments/assets/a699cfdd-d82e-4815-87f5-f3333746d4db" />

```
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
y

```

<img width="1086" height="623" alt="image" src="https://github.com/user-attachments/assets/aaa7dbc2-cbb7-46de-b891-292578ac207d" />

<img width="1405" height="692" alt="image" src="https://github.com/user-attachments/assets/9124e589-7f66-4a57-bdd6-da8330e93b26" />

```

from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)

x_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(x_train,lr.predict(x_train),color='red')
lr.coef_
lr.intercept_

```

<img width="1122" height="721" alt="image" src="https://github.com/user-attachments/assets/18a011ea-fc95-4f76-a2ba-4a7af44a2ab5" />

```

y_pred=lr.predict(x_test)
mse=mean_squared_error(Y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("MSE : ",mse)
print("RMSE : ",rmse)
print("MAE : ",mae)
print("R2 : ",r2)

```

<img width="807" height="156" alt="image" src="https://github.com/user-attachments/assets/b0b745e9-6708-4558-9723-eddca91bf428" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
