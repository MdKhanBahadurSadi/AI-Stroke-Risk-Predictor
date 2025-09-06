import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_and_save_model():
    """
    This function loads the stroke dataset, preprocesses the data,
    trains a Random Forest Classifier, and saves the trained model
    and the LabelEncoders to .pkl files.
    """
    try:
        file_path = 'healthcare-dataset-stroke-data.csv'
        df = pd.read_csv(file_path)
        df = df.drop('id', axis=1)

        df['bmi'] = df['bmi'].replace('N/A', np.nan)
        df['bmi'] = pd.to_numeric(df['bmi'])
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        df = df[df['gender'] != 'Other']

        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        encoders = {}
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            encoders[feature] = le

        X = df.drop('stroke', axis=1)
        y = df['stroke']

        # Train a Random Forest model on the entire dataset
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X, y)

        # Save the trained model to a pickle file
        with open('stroke_model.pkl', 'wb') as model_file:
            pickle.dump(rf_model, model_file)

        # Save the encoders to a pickle file
        with open('label_encoders.pkl', 'wb') as encoders_file:
            pickle.dump(encoders, encoders_file)

        print("Model and encoders saved successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_and_save_model()
