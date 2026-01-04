import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Visualization")

st.title("Rectified Linear Unit (ReLU)")

x = np.linspace(-10, 10, 100)

# ReLU outputs x if x > 0, otherwise it outputs 0
y = np.maximum(0, x)

#formula
st.latex(r"f(x) = \max(0, x)")

fig, ax = plt.subplots()
ax.plot(x, y, color='blue', linewidth=2, label='ReLU')
ax.set_title("ReLU Activation")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)