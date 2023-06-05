# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('E:\Kuliah\Semester 4\AI\Lung Cancer\cancer patient data sets.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('E:\Kuliah\Semester 4\AI\Lung Cancer\cancer patient data sets.csv')
data

selected_columns = ['Age', 'Gender', 'Air Pollution', 'Genetic Risk', 'chronic Lung Disease',
                    'Smoking', 'Passive Smoker', 'Shortness of Breath', 'Level']
selected_data = data.loc[:, selected_columns]

# convert column Level(dependent variable) from category to int
selected_data['Level'].unique()
## Now, apply LabelEncoding for this Level column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
selected_data['Level'] = le.fit_transform(selected_data['Level'])

## Spliting the dataset into training and testing
X = selected_data.iloc[:, :-1].values  # other columns
y = selected_data.iloc[:, -1].values  # level
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# Feature Scaling of dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)