import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title("🏇 Horse Racing Predictor")

uploaded_file = st.file_uploader("Upload race Excel file", type=["xlsx"])

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df['Odds'] = df['Odds'].replace('scratch', np.nan).astype(float)
    df['Implied_Prob'] = 1 / df['Odds']
    num_features = [
        'Odds', 'Jockey Win %', 'Trainer Win %', 'Post Position',
        'Age', 'Weight', 'Days Off', 'Prime Power',
        'Last Class', 'Average Class', 'Early Pace 1', 'Early Pace 2',
        'Late Pace', 'Average Speed', 'Best Speed', 'Average Distance'
    ]
    cat_features = ['Jockey', 'Trainer', 'Owner', 'Medication',
                    'Run Style', 'Sire', 'Dam', 'Sex']

    df['Odds_JockeyWin'] = df['Odds'] * df['Jockey Win %']
    df['Post_RunStyle'] = df['Post Position'].astype(str) + "_" + df['Run Style']
    df['Pace_Ratio'] = (df['Early Pace 1'] + df['Early Pace 2']) / df['Late Pace'].replace(0, 1)
    df['Form_Momentum'] = df['Recent Form'].apply(lambda x: np.mean([int(c) for c in str(x) if c.isdigit()]) if pd.notnull(x) else np.nan)
    df['Class_Adjusted_Speed'] = df['Average Speed'] - df['Average Class']
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    X = preprocessor.fit_transform(df)
    return X, df[['Race Number', 'Horse Number']]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.model(x)

def load_nn_model(input_dim, output_dim):
    model = MLP(input_dim, output_dim)
    model.load_state_dict(torch.load("nn_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    X, ids = preprocess_data(df)

    st.markdown("### Choose model for prediction")
    model_type = st.radio("Model", ["Logistic Regression", "XGBoost", "Neural Network"])

    if model_type == "Logistic Regression":
        model = joblib.load("logistic_model.pkl")
        preds = model.predict_proba(X)

    elif model_type == "XGBoost":
        model = joblib.load("xgb_model.pkl")
        preds = model.predict_proba(X)

    else:
        nn_model = load_nn_model(X.shape[1], len(np.unique(df['Finish Position'])))
        with torch.no_grad():
            preds = nn_model(torch.tensor(X.toarray() if hasattr(X, 'toarray') else X, dtype=torch.float32)).numpy()

    prob_df = pd.DataFrame(preds, columns=[f"P({i+1}st)" for i in range(preds.shape[1])])
    result_df = pd.concat([ids.reset_index(drop=True), prob_df], axis=1)

    st.write("### Prediction Results")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
