import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Visualization")

#sigmoid info
with st.expander("See Mathematical Properties"):
    st.latex(r"f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}")
    st.write("""
    * **Squashing:** Maps inputs to a (0, 1) probability range.
    * **Smooth Gradient:** The curve is smooth, no sharp corners.
    * **Vanishing Gradient:** Notice how flat the curve gets at $x > 5$ or $x < -5$. In these regions, the gradient is almost zero, which stops the neural network from learning effectively.
    """)

st.write("Test your own number below:")
user_input = st.number_input("Enter a value for x:", value=0.0, step=0.5)
user_output = 1 / (1 + np.exp(-user_input))
st.write(f"**Result:** When input $x = {user_input}$, output $y = {user_output:.4f}$")

x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots()
ax.plot(x, y, color='green', linewidth=2, label='Sigmoid Curve')
ax.scatter([user_input], [user_output], color='red', zorder=5, label='Your Point')

ax.set_title("Sigmoid Activation")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (y)")
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)