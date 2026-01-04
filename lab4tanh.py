import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Visualization")

#yanh info
with st.expander("See Mathematical Properties"):
    st.latex(r"f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
    st.write("""
    * **Zero-Centered:** Unlike Sigmoid, this outputs values between -1 and 1. This helps the model learn faster because the average output is closer to 0.
    * **Steeper Gradient:** The curve is steeper than Sigmoid around 0, meaning the gradients are stronger.
    * **Vanishing Gradient:** Similar to Sigmoid, it suffers from flat curves at high or low numbers, causing learning to slow down.
    """)
# -------------------------------------------------------------

st.write("Test your own number below:")
user_input = st.number_input("Enter a value for x:", value=0.0, step=0.5)
user_output = np.tanh(user_input)
st.write(f"**Result:** When input $x = {user_input}$, output $y = {user_output:.4f}$")

x = np.linspace(-10, 10, 100)
y = np.tanh(x)

fig, ax = plt.subplots()
ax.plot(x, y, color='red', linewidth=2, label='Tanh Curve')
ax.scatter([user_input], [user_output], color='red', zorder=5, label='Your Point')

ax.set_title("Tanh Activation")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)