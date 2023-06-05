import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('cancer patient data sets.csv')

# Preprocess the dataset
selected_columns = ['Age', 'Gender', 'Air Pollution', 'Genetic Risk', 'chronic Lung Disease',
                    'Smoking', 'Passive Smoker', 'Shortness of Breath', 'Level']
selected_data = data.loc[:, selected_columns]

# Convert 'Level' column from category to int using LabelEncoder
le = LabelEncoder()
selected_data['Level'] = le.fit_transform(selected_data['Level'])

# Create the Streamlit app
def main():
    st.title('Prediksi Resiko Kanker Paru-Paru')

    # Sidebar inputs
    st.sidebar.header('User Inputs')
    age = st.sidebar.number_input('Age', min_value=1, max_value=100, value=50)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    air_pollution = st.sidebar.number_input('Air Pollution', min_value=1, max_value=10, value=5)
    genetic_risk = st.sidebar.number_input('Genetic Risk', min_value=1, max_value=10, value=5)
    chronic_lung_disease = st.sidebar.number_input('Chronic Lung Disease', min_value=1, max_value=10, value=1)
    smoking = st.sidebar.number_input('Smoking', min_value=1, max_value=10, value=1)
    passive_smoker = st.sidebar.number_input('Passive Smoker', min_value=1, max_value=10, value=1)
    shortness_of_breath = st.sidebar.number_input('Shortness of Breath', min_value=1, max_value=10, value=1)

    # Submit button
    submitted = st.sidebar.button('Submit')

    if submitted:
        # Convert gender input to numeric value
        gender_numeric = 1 if gender == 'Male' else 2

        # Create a DataFrame for the user input
        user_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_numeric],
            'Air Pollution': [air_pollution],
            'Genetic Risk': [genetic_risk],
            'chronic Lung Disease': [chronic_lung_disease],
            'Smoking': [smoking],
            'Passive Smoker': [passive_smoker],
            'Shortness of Breath': [shortness_of_breath]
        })

        # Compare user input with the dataset
        user_gender = user_data['Gender'].values[0]
        user_age = user_data['Age'].values[0]
        filtered_data = selected_data[(selected_data['Gender'] == user_gender) & (selected_data['Age'] == user_age)]

        if filtered_data.empty:
            st.subheader('Risk Level Prediction')
            st.write('No matching data found in the dataset.')

            # Append user input as new data to the dataset
            new_data = user_data.copy()
            new_data['Level'] = 'Unknown'
            updated_data = pd.concat([selected_data, new_data], ignore_index=True)
            updated_data.to_csv('updated_cancer_patient_data.csv', index=False)
            st.write('User input has been added to the dataset for further learning.')
        else:
            risk_level = le.inverse_transform(filtered_data['Level'].values)[0]
            st.subheader('Prediksi Resiko Terkena Kanker Paru-Paru')
            st.write('Berdasarkan Data, Prediksi Resiko Anda terkena kanker adalah:', risk_level)


if __name__ == '__main__':
    main()
