import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


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
    st.sidebar.header('Data Inputan')
    age = st.sidebar.number_input('Umur', min_value=1, max_value=100, value=33)
    gender = st.sidebar.selectbox('Jenis Kelamin', ['Laki-Laki', 'Perempuan'])
    air_pollution = st.sidebar.number_input('Polusi Udara', min_value=1, max_value=10, value=2)
    genetic_risk = st.sidebar.number_input('Resiko Genetika', min_value=1, max_value=10, value=3)
    chronic_lung_disease = st.sidebar.number_input('Penyakit Kronis Paru-Paru', min_value=1, max_value=10, value=2)
    smoking = st.sidebar.number_input('Perokok Aktif', min_value=1, max_value=10, value=3)
    passive_smoker = st.sidebar.number_input('Perokok Pasif', min_value=1, max_value=10, value=2)
    shortness_of_breath = st.sidebar.number_input('Sesak Nafas', min_value=1, max_value=10, value=2)

    # Submit button
    submitted = st.sidebar.button('Prediksi')

    

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
