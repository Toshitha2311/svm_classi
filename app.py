import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="SVM Classifier", layout="wide")

st.title("🟢 SVM Classification Model (FORCED FIX)")

# ----------------------------
# DATA SOURCE
# ----------------------------
option = st.radio("Choose Dataset Source", ["Default (Iris)", "Upload CSV"])

if option == "Default (Iris)":
    iris = load_iris(as_frame=True)
    df = iris.frame
else:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.stop()
    df = pd.read_csv(file)

# ----------------------------
# PREVIEW
# ----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# TARGET
# ----------------------------
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

# ----------------------------
# 🔥 FORCE FIX TARGET (IMPORTANT)
# ----------------------------

# If numeric continuous → convert into classes using bins
if y.dtype != "object" and y.nunique() > 10:
    st.warning("⚠ Continuous target detected → converting into classification bins")
    y = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])

# If still numeric but small unique values → keep as class
y = y.astype(str)

# ----------------------------
# SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# SCALING
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# MODEL
# ----------------------------
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ----------------------------
# METRICS
# ----------------------------
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.success(f"Accuracy: {acc:.2f}")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ----------------------------
# PREDICTION
# ----------------------------
st.subheader("Make Prediction")

input_data = []
for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    input_data.append(val)

input_array = scaler.transform(np.array(input_data).reshape(1, -1))
prediction = model.predict(input_array)

st.success(f"Prediction: {prediction[0]}")