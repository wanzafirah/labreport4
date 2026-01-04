import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Visualization")

#info relu
with st.expander("See Mathematical Properties"):
    st.latex(r"f(x) = \max(0, x)")
    st.write("""
    * **Non-Linearity:** It allows the model to learn complex patterns despite looking like a linear line.
    * **Sparsity:** For any negative input ($x < 0$), the output is exactly 0. This "turns off" neurones, making the network lighter and efficient.
    * **No Vanishing Gradient (for positive x):** unlike Sigmoid/Tanh, the gradient is always 1 for positive numbers, allowing deep networks to learn faster.
    """)


st.write("Test your own number below:")
user_input = st.number_input("Enter a value for x:", value=2.0, step=0.5)
user_output = max(0, user_input)
st.write(f"**Result:** When input $x = {user_input}$, output $y = {user_output}$")

x = np.linspace(-10, 10, 100)
y = np.maximum(0, x)

fig, ax = plt.subplots()
ax.plot(x, y, color='blue', linewidth=2, label='ReLU Curve')
ax.scatter([user_input], [user_output], color='red', zorder=5, label='Your Point')
ax.vlines(user_input, 0, user_output, colors='red', linestyles='dashed', alpha=0.5)

ax.set_title("ReLU Activation")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)