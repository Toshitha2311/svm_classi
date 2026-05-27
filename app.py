import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Decision Tree Regressor", layout="wide")

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/decision_tree_regressor.pkl"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

st.title("🌳 Decision Tree Regression Model")

# ---------------- DEFAULT DATASET ----------------
@st.cache_data
def load_default_data():
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    df.to_csv(DATA_PATH, index=False)

    return df

# ---------------- DATA SOURCE ----------------
choice = st.radio("📌 Choose Dataset", ["Default Dataset", "Upload CSV"])

df = None

if choice == "Upload CSV":
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df.to_csv(DATA_PATH, index=False)
else:
    df = load_default_data()

# ---------------- MAIN APP ----------------
if df is not None:

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Target Column", df.columns, index=len(df.columns)-1)

    feature_cols = st.multiselect(
        "📌 Feature Columns",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col]
    )

    if feature_cols and target_col:

        X = df[feature_cols]
        y = df[target_col]

        test_size = st.slider("🧪 Test Size", 0.1, 0.5, 0.2)

        max_depth = st.slider("🌳 Max Depth", 1, 20, 5)

        if st.button("🚀 Train Model"):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model = DecisionTreeRegressor(
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # ---------------- SAVE MODEL ----------------
            joblib.dump(model, MODEL_PATH)

            st.success("💾 Model saved successfully!")

            # ---------------- METRICS ----------------
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success(f"📉 MSE: {mse:.4f}")
            st.success(f"📈 R² Score: {r2:.4f}")

            # ---------------- ACTUAL VS PREDICTED ----------------
            st.subheader("📊 Actual vs Predicted")

            fig, ax = plt.subplots()

            ax.scatter(y_test, y_pred)
            ax.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                color="red"
            )

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

            st.pyplot(fig)

            # ---------------- RESIDUALS ----------------
            st.subheader("📉 Residual Plot")

            residuals = y_test - y_pred

            fig2, ax2 = plt.subplots()

            ax2.scatter(y_pred, residuals)
            ax2.axhline(0, color="red")

            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Residuals")

            st.pyplot(fig2)

            # ---------------- FEATURE IMPORTANCE ----------------
            st.subheader("📊 Feature Importance")

            importance = model.feature_importances_

            fig3, ax3 = plt.subplots()

            ax3.barh(feature_cols, importance)
            ax3.set_xlabel("Importance")

            st.pyplot(fig3)

            st.session_state["features"] = feature_cols

    # ---------------- PREDICTION ----------------
    if os.path.exists(MODEL_PATH):

        st.sidebar.header("🔮 Prediction Panel")

        model = joblib.load(MODEL_PATH)

        features = st.session_state.get("features", [])

        inputs = []

        for col in features:
            inputs.append(
                st.sidebar.number_input(col, value=0.0)
            )

        if st.sidebar.button("Predict"):

            input_data = np.array(inputs).reshape(1, -1)

            prediction = model.predict(input_data)

            st.sidebar.success(f"📈 Predicted Value: {prediction[0]:.4f}")

else:
    st.warning("📌 Please load dataset")