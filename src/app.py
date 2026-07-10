from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Detector de Billetes",
    page_icon="💵",
    layout="centered"
)

model_path = Path(__file__).resolve().parent.parent / "models" / "banknote_model.pkl"
model = joblib.load(model_path)

st.title("💵 Detector de Billetes")
st.write(
    "Ingresa las características del billete para determinar si es auténtico o falso."
)

variance = st.number_input("Variance", format="%.5f")
skewness = st.number_input("Skewness", format="%.5f")
curtosis = st.number_input("Curtosis", format="%.5f")
entropy = st.number_input("Entropy", format="%.5f")

if st.button("Predecir"):
    input_data = pd.DataFrame(
        [[variance, skewness, curtosis, entropy]],
        columns=["variance", "skewness", "curtosis", "entropy"]
    )

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("El billete es auténtico.")
    else:
        st.error("El billete es falso.")

st.caption(
    "Recursos utilizados: documentación oficial de Streamlit, pandas y joblib."
)