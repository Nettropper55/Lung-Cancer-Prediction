import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('E:\Kuliah\Semester 4\AI\Lung Cancer\cancer patient data sets.csv')

selected_columns = ['Age', 'Gender', 'Air Pollution', 'Genetic Risk', 'chronic Lung Disease',
                    'Smoking', 'Passive Smoker', 'Shortness of Breath', 'Level']
selected_data = data.loc[:, selected_columns]

# Convert column Level (dependent variable) from category to int
le = LabelEncoder()
selected_data['Level'] = le.fit_transform(selected_data['Level'])

## Split the dataset into training and testing
X = selected_data.iloc[:, :-1].values  # other columns
y = selected_data.iloc[:, -1].values  # level

sc = StandardScaler()
X_train = sc.fit_transform(X)

lr = LogisticRegression()
lr.fit(X_train, y)

# Save the trained model
joblib.dump(lr, 'trained_model.pkl')

# Rest of the code remains the same
