# Importing modules
import pandas as pd
import numpy as np
import pickle 

# Importing data set
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# Customizing data frame for target Malignant(Cancer)=0 & Benign(Non Cancer)=1
df = pd.DataFrame (np.c_[cancer['data'],cancer['target']], columns = np.append (cancer['feature_names'], ['target']))
df.head()

# To train a Support Vector Machine classifier
from sklearn.svm import SVC
svmc = SVC(kernel='linear')

# Splitting the data into 'Training' and 'Testing' datasets
from sklearn.model_selection import train_test_split
df_train , df_test = train_test_split(df, test_size = 0.2 , random_state = 20 )

# Training the model using the training dataset"
svmc.fit(df_train[['mean radius','mean concavity']],df_train['target'])

# Passing test inputs for trained model
#INPUT_ARRAY = [13.64,0.01857]
INPUT_ARRAY = [18.22,0.1772]

# predict test set using the trained model
y_predict = svmc.predict([INPUT_ARRAY])

# Show prediction
print (y_predict)

# Saving the trained model
from joblib import dump, load
dump(svmc, 'svc_model.model')
