import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel('1553768847_housing.xlsx') 

#reading few rows
data.head()

#input and ouput
p= data.iloc[:,:-1].values
q= data.iloc[:,9].values

#missing values
from sklearn.preprocessing import Imputer
missingVal = Imputer( missing_values="NaN", strategy="mean", axis=0)
p[:,0:8] = missingVal.fit_transform(p[:,0:8])

#categorical data
from sklearn.preprocessing import LabelEncoder
p_label = LabelEncoder()
p[:,8] = p_label.fit_transform(p[:,8])

#x_label.classes_

#test and train data
from sklearn.model_selection import train_test_split
p_train,p_test,q_train,q_test = train_test_split(p,q,test_size=0.2,
                                                 random_state=0)

#standardize data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
p_train = scale.fit_transform(p_train)
p_test = scale.fit_transform(p_test)

#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(p_train,q_train)

predictedVal = lr.predict(p_test)
predictedVal2 = lr.predict(p_train)
#checking accuracy
lr.score(p_train,q_train) # 0.6378966554411029
lr.score(p_test,q_test)   # 0.6253939067197227

# rmse
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(q_test,predictedVal)) # 69890.58962190864
np.sqrt(mean_squared_error(q_train,predictedVal2)) # 69615.2854305203

#decision tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(p_train,q_train)

predictedVal3 = dtr.predict(p_test)
predictedVal4 = dtr.predict(p_train)
dtr.score(p_train,q_train) # 1.0
dtr.score(p_test,q_test) # 0.5446154120725679

#rmse
np.sqrt(mean_squared_error(q_test,predictedVal3)) # 77058.48206826876
np.sqrt(mean_squared_error(q_train,predictedVal4)) # 0.0

#random forest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(p_train,q_train)

predictedVal5 = rfr.predict(p_test)
predictedVal6 = rfr.predict(p_train)
rfr.score(p_train,q_train) # 0.9630174595597636
rfr.score(p_test,q_test) # 0.7277729361300703
#rmse
np.sqrt(mean_squared_error(q_test,predictedVal5)) # 59579.511115479356
np.sqrt(mean_squared_error(q_train,predictedVal6)) # 22247.781493357117


#bonus exercise : extracting median income
d1 = data.drop("median_income",axis=1)
d2 = data.drop(d1,axis=1)

p_train2,p_test2,q_train2,q_test2 = train_test_split(d2,q,
                                                     test_size=0.2)

lr2 = LinearRegression()
lr2.fit(p_train2,q_train2)

predictedVal7 = lr2.predict(p_test2)
predictedVal8 = lr2.predict(p_train2)

#visualizing
plt.scatter(q_train,predictedVal8,color='blue',s=5)
plt.scatter(q_test2,predictedVal7,color='red',s=5)
plt.xlabel('Actual median values')
plt.ylabel('Predicted median values')
plt.show()















