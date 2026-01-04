import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Visualization")

st.title("Hyperbolic Tangent (Tanh)")

x = np.linspace(-10, 10, 100)

# Tanh is similar to Sigmoid but zero-centered
y = np.tanh(x)

# formula
st.latex(r"f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")

fig, ax = plt.subplots()
ax.plot(x, y, color='red', linewidth=2, label='Tanh')
ax.set_title("Tanh Activation")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)