import pandas as pd
import numpy as np
df = pd.read_csv("Microsoft_code_fun_do/flood_dataset.csv")
df.head()
from sklearn import linear_model
logreg = linear_model.LogisticRegression()
X= df[["PLACEID","MONTH"]]
y=df[["TARGET(2018_month)"]]
logreg.fit(X,y)
#print(logreg.coef_)
#print(logreg.intercept_)
result = logreg.predict_proba([[1,10]])
t = result[0][1]
t
