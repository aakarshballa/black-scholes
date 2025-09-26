# app_streamlit.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from math_1 import build_grid

st.set_page_config(page_title="Black–Scholes Moneyness", layout="wide")
st.title("Black–Scholes Moneyness Explorer")

with st.sidebar:
    st.header("Inputs")
    S = st.number_input("Spot S", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    r = st.number_input("Risk-free r (annual, %)", value=3.50, step=0.10, format="%.2f") / 100
    q = st.number_input("Dividend q (annual, %)", value=0.00, step=0.10, format="%.2f") / 100
    sigma = st.number_input("Volatility σ (annual, %)", value=25.0, min_value=0.01, step=0.25, format="%.2f") / 100
    T = st.number_input("Time to expiry T (years)", value=0.5, min_value=0.0001, step=0.05, format="%.4f")
    opt_type = st.selectbox("Option type", ["call", "put"])
    st.header("Strike Grid")
    m_lo = st.number_input("Min moneyness (K/S)", value=0.7, min_value=0.1, step=0.05, format="%.2f")
    m_hi = st.number_input("Max moneyness (K/S)", value=1.3, min_value=0.1, step=0.05, format="%.2f")
    step = st.number_input("Strike step", value=2.5, min_value=0.1, step=0.1, format="%.2f")
    atm_tol = st.number_input("ATM tolerance (±%)", value=1.0, min_value=0.1, step=0.1, format="%.1f") / 100

df = build_grid(S, r, q, sigma, T, opt_type, m_lo, m_hi, step, atm_tol)
st.subheader("Option Grid")
st.dataframe(df, use_container_width=True)

st.subheader("Price vs Strike")
fig1, ax1 = plt.subplots()
ax1.plot(df["Strike (K)"], df["BS Price"])
ax1.set_xlabel("Strike (K)")
ax1.set_ylabel("Black–Scholes Price")
ax1.set_title("Price vs Strike")
st.pyplot(fig1)

st.subheader("Greeks vs Strike")
greek = st.selectbox("Greek", ["Delta", "Gamma", "Vega", "Theta (per year)", "Rho"])
fig2, ax2 = plt.subplots()
ax2.plot(df["Strike (K)"], df[greek])
ax2.set_xlabel("Strike (K)")
ax2.set_ylabel(greek)
ax2.set_title(f"{greek} vs Strike")
st.pyplot(fig2)
