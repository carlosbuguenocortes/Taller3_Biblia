import streamlit as st
import requests

st.title("Dashboard Biblia")

data = requests.get(
    "http://127.0.0.1:8000/dashboard"
).json()

st.metric(
    "Versículos",
    data["total_versiculos"]
)

st.metric(
    "Libros",
    data["total_libros"]
)

st.metric(
    "Capítulos",
    data["total_capitulos"]
)
