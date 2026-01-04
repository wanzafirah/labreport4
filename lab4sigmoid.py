import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Visualization")

st.title("Sigmoid Function")

x = np.linspace(-10, 10, 100)

# Sigmoid 'squashes' the input between 0 and 1
y = 1 / (1 + np.exp(-x))

# formula
st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")

fig, ax = plt.subplots()
ax.plot(x, y, color='green', linewidth=2, label='Sigmoid')
ax.set_title("Sigmoid Activation")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)