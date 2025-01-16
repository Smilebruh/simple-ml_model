import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve

dataset = pd.read_csv("diabetes.csv")
corr = dataset.corr()

pl = sns.PairGrid(dataset, hue="Outcome", palette="tab10")
pl.map_upper(sns.scatterplot)
pl.map_diag(sns.kdeplot)

sns.heatmap(corr.round(2),annot=True,cmap="vlag",vmax=1,vmin=0)

X  = dataset.iloc[:,:7].to_numpy()

y = dataset['Outcome'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True,test_size=0.2)

Model2 = LogisticRegression(max_iter=300)
Model2.fit(X_train, y_train)

y_pred = Model2.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
roc = roc_curve(y_pred,y_test)
print(f"roc = {roc}")
print(f"accuracy = {accuracy}")

with open('model-diabetes.pkl',"wb") as file:
    pickle.dump(Model2,file)
    file.close()