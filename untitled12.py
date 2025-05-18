import pandas as pd

df = pd.read_csv('DiseaseAndSymptoms.csv')

df.head()

df.info()

# prompt: remove the columns from symptom_4 to symptom_17

cols_to_drop = [f'Symptom_{i}' for i in range(4, 18)]
df = df.drop(columns=cols_to_drop)
df.head()
df.info()

# prompt: duplicates

# Count duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

# Display duplicate rows
if duplicate_rows > 0:
    print("\nDuplicate rows:")
    print(df[df.duplicated(keep=False)]) # keep=False marks all duplicates as True

# Remove duplicate rows
df_no_duplicates = df.drop_duplicates()
print(f"\nShape of DataFrame after removing duplicates: {df_no_duplicates.shape}")

df.head()

# prompt: Encode the column symptoms

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
df['Symptoms'] = df[['Symptom_1', 'Symptom_2', 'Symptom_3']].apply(lambda row: [s for s in row if pd.notna(s)], axis=1)
encoded_symptoms = mlb.fit_transform(df['Symptoms'])

# Create new columns for each symptom
symptom_cols = [f'Symptom_{i+1}' for i in range(encoded_symptoms.shape[1])]
encoded_df = pd.DataFrame(encoded_symptoms, columns=symptom_cols)

# Concatenate the original dataframe with the encoded symptoms dataframe
df = pd.concat([df.drop(columns=['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptoms']), encoded_df], axis=1)

df.head()
df.info()

# prompt: train and test the dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df.drop('Disease', axis=1)
y = df['Disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a model (Logistic Regression as an example)
model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print classification report for more detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



import streamlit as st


st.title("AI Health Assistant")
symptoms = st.multiselect("Select your symptoms", X.columns)
input_data = [1 if s in symptoms else 0 for s in X.columns]

if st.button("Predict"):
    prediction = model.predict([input_data])
    st.success(f"Possible diagnosis: {prediction[0]}")



