import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

@st.cache_data
def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

def preprocess_data(df):
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Binary encode Yes/No and gender columns
    df['gender'] = df['gender'].map({'Male':1, 'Female':0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
        df[col] = df[col].map({'Yes':1, 'No':0})

    # One-hot encoding for remaining categorical variables
    df = pd.get_dummies(df, drop_first=True)
    return df

@st.cache_resource
def train_models(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=101)
    xgb.fit(X_train, y_train)
    
    return rf, xgb

def main():
    st.title("Customer Churn Prediction: Random Forest vs XGBoost")

    df = load_data()
    df = preprocess_data(df)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    rf_model, xgb_model = train_models(X_train, y_train)

    st.subheader("Model Comparison")

    # Predict with both models
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)

    # Accuracy scores
    rf_acc = accuracy_score(y_test, rf_preds)
    xgb_acc = accuracy_score(y_test, xgb_preds)

    st.write(f"**Random Forest Accuracy:** {rf_acc:.4f}")
    st.write(f"**XGBoost Accuracy:** {xgb_acc:.4f}")

    # Classification reports
    st.write("### Random Forest Classification Report")
    st.text(classification_report(y_test, rf_preds))

    st.write("### XGBoost Classification Report")
    st.text(classification_report(y_test, xgb_preds))

    # Optionally: confusion matrices or feature importances here

if __name__ == "__main__":
    main()
# Run the Streamlit app
# To run the app, use the command: streamlit run app/streamlit_app.py