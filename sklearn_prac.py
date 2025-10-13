from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot
matplotlib.use("TkAgg")   # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



X,y = fetch_california_housing(return_X_y=True)

mod = KNeighborsRegressor()
pipe = Pipeline([
                 ("scale" , StandardScaler()),
                 ("model",KNeighborsRegressor(n_neighbors=1))
])


##print(pipe.get_params())
mod = GridSearchCV(estimator=pipe,param_grid={'model__n_neighbors' : [1,2,3,4,5,6,7,8,9,10]},cv=3)

mod.fit(X , y)
data = pd.DataFrame(mod.cv_results_)
print(data)
pred = mod.predict(X)
print(pred)

plt.scatter(pred,y)
plt.show()