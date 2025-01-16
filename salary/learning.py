import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

dataset = pd.read_csv("Salary_dataset.csv")

X = dataset["YearsExperience"]
y = dataset["Salary"]

X_train,X_test,y_train,y_test = train_test_split(X,y)

plt.scatter(X_train,y_train)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

Model = LinearRegression()
Model.fit(np.array(X_train).reshape(-1,1),np.array(y_train).reshape(-1,1))

y_pred = Model.predict(np.array(X_test).reshape(-1,1))

plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred)
plt.show()

with open("my_model.pkl","wb") as model:
    pickle.dump(Model,model)
    model.close()
