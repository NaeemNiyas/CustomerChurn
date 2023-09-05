import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import openpyxl
 

data = pd.read_excel("churnpredict/customer_churn_large_dataset.xlsx")
# checking for missing values

data.drop(['CustomerID', 'Name'], axis=1, inplace=True)

lb1=LabelEncoder()

lb2=LabelEncoder()

data['Gender']=lb1.fit_transform(data['Gender'])
data['Location']=lb2.fit_transform(data['Location'])

df = data[[ 'Age','Gender','Location',	'Subscription_Length_Months','Monthly_Bill','Total_Usage_GB','Churn']]

#create feature set and labels
X = df.drop(columns='Churn',axis=1)
y = df['Churn']
#train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=100)
#building the model & printing the score
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)
print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))

"""
input_data = (376,3,2,29,4,115046,4,1,0,119346)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#prediction = model.predict(input_data_reshaped)
prediction = xgb_model.predict(input_data_reshaped)
print(prediction[0])
"""

pickle.dump(xgb_model, open('xgb_model.pkl','wb')) 
 