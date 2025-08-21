# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kanigavel M
RegisterNumber:  212224240070
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Create Dataset
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Marks': [10,20,30,40,50,60,65,75,85,95]
}
df = pd.DataFrame(data)

# Step 2: Display head and tail
print("HEAD VALUES:\n", df.head())
print("\nTAIL VALUES:\n", df.tail())

# Step 3: Define Independent (X) and Dependent (y)
X = df[['Hours']]
y = df['Marks']

print("\nX VALUES:\n", X.values)
print("\nY VALUES:\n", y.values)

# Step 4: Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTRAINING SET (X_train):\n", X_train.values)
print("\nTESTING SET (X_test):\n", X_test.values)

# Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

print("\nACTUAL VALUES:", list(y_test.values))
print("PREDICTED VALUES:", list(y_pred))

# Step 7: Errors
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMSE =", mse)
print("MAE =", mae)
print("RMSE =", rmse)

# Step 8A: Training Graph
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Hours of Study")
plt.ylabel("Marks Scored")
plt.title("Training Set - Simple Linear Regression")
plt.legend()
plt.show()

# Step 8B: Testing Graph
plt.scatter(X_test, y_test, color="green", label="Testing Data (Actual)")
plt.scatter(X_test, y_pred, color="orange", marker="x", s=100, label="Predicted Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Hours of Study")
plt.ylabel("Marks Scored")
plt.title("Testing Set - Simple Linear Regression")
plt.legend()
plt.show()
```

## Output:
## Output:
### HEAD VALUES
![Screenshot_21-8-2025_144039_localhost](https://github.com/user-attachments/assets/7f4916fb-f27d-46c2-8460-99f337c5344e)


## TAIL VALUES
![Screenshot_21-8-2025_144111_localhost](https://github.com/user-attachments/assets/4e7bf020-51c5-4404-8bc0-4e2d6b254e18)


## X VALUES 
![Screenshot_21-8-2025_144147_localhost](https://github.com/user-attachments/assets/41a6fc12-2078-46d4-bddb-1c1346a7624e)

## Y VALUES
 ![Screenshot_21-8-2025_14426_localhost](https://github.com/user-attachments/assets/21c7e88c-cf4c-4687-ae7f-41fa1c51a97e)


## PREDICTED VALUE 



![Screenshot_21-8-2025_144319_localhost](https://github.com/user-attachments/assets/5c02492e-3804-4f9e-87ad-1161983d3088)


## ACTUAL VALUES



![Screenshot_21-8-2025_144359_localhost](https://github.com/user-attachments/assets/3ff4be1e-98ca-4a2f-bd81-473c183c1f94)


## TRAINING SET

![Screenshot_21-8-2025_144435_localhost](https://github.com/user-attachments/assets/b2ea4f83-1f72-41fa-b773-f0ebfc73f012)

<img width="563" height="453" alt="ml2 1" src="https://github.com/user-attachments/assets/7ae67899-8776-40be-b4a7-8f0ac4d8f2b2" />


## TESTING SET

![Screenshot_21-8-2025_144458_localhost](https://github.com/user-attachments/assets/b3fccf39-a6fc-4d5c-b882-3e3d21b50ef0)

<img width="563" height="453" alt="ex 2 2" src="https://github.com/user-attachments/assets/6f4518e8-c28f-45a5-aef2-8da1230efdc9" />


## MSE, MAE and RMSE


![Screenshot_21-8-2025_144533_localhost](https://github.com/user-attachments/assets/17667fe7-6336-41ee-8163-80695b24e3d7)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
