import streamlit as st
import numpy as np
import plotly.graph_objs as go  # Import plotly graph objects
import pandas as pd

# Define the Jacobian matrix and calculate eigenvalues
def UL_eigen (A, iters= 5000,tol = 1e-15):
    m,n = A.shape 
    I = np.identity (np.shape(A)[0])
    shift_A = pd.shift(A) + 1
    A = A + I * (shift_A)
    
    D1 = A ; D2 = np.ones(np.shape(A))
    k=0
    i=0
    
    while (np.allclose(np.diagonal (D1), np.diagonal (D2), tol)==False) and i<=iters:
        L,U = lud(D1)
        D2 = np.matmul (U,L)
        
        if (np.allclose(np.diagonal (D1), np.diagonal (D2), tol)==True):
            return np.round(np.diagonal(D2) -(shift_A),3)
            
        D1 = D2
        D2 = np.zeros((m,n))
        i+=1

def jacobian_eigenvalues():
    A = np.array([[0, 1], [0, 0]])  # Jacobian matrix for this system
    eigenvalues = UL_eigen(A)
    return eigenvalues

# Display equations used in the model
def display_equations():
    st.write("### Equations Used")
    st.latex(r"""
        \frac{d^2u}{dy^2} = -P
    """)
    st.write("Converted into first-order differential equations:")
    st.latex(r"""
        u_1' = u_2
    """)
    st.latex(r"""
        u_2' = -P
    """)
    st.write("Analytical solution:")
    st.latex(r"""
        u(y) = y + \frac{P}{2} \cdot y \cdot (1 - y)
    """)
    st.write("BVP Finite Difference Method:")
    st.latex(r"""
        A u = b
    """)
    st.latex(r"""
        \mathbf{A} \mathbf{u} = \mathbf{b}
    """)
    st.write("In matrix form:")
    st.latex(r"""
        \mathbf{u}' = \mathbf{A} \mathbf{u} + \mathbf{b}
    """)
    st.write("The Jacobian is matrix A")
    st.latex(r"""
        \mathbf{A} = \begin{bmatrix}
        0 & 1 \\
        0 & 0 \\
        \end{bmatrix}, \quad
        \mathbf{u} = \begin{bmatrix}
        u_1 \\
        u_2 \\
        \end{bmatrix}, \quad
        \mathbf{b} = \begin{bmatrix}
        0 \\
        -P \\
        \end{bmatrix}
    """)

# Define the analytical solution function
def analytical_solution(y, P):
    return y + (P / 2) * y * (1 - y)

# Define the BVP solver function
def solve_couette_poiseuille(P, N):
    dy = 1.0 / (N - 1)
    y = np.linspace(0, 1, N)
    A = np.zeros((N, N))
    b = np.zeros(N)
    for i in range(1, N - 1):
        A[i, i - 1] = 1
        A[i, i] = -2
        A[i, i + 1] = 1
        b[i] = -P * dy**2
    A[0, 0] = 1
    b[0] = 0
    A[-1, -1] = 1
    b[-1] = 1
    u = np.linalg.solve(A, b)
    return y, u

# Define the IVP solvers for Explicit and Implicit Euler methods
def IVP_Explicit(p, h):
    iteration = round(1/h) + 1
    u1 = np.zeros(iteration)
    u2 = np.zeros(iteration)
    u1[0] = 0
    u2[0] = initial_value(p)
    for i in range(iteration - 1):
        u1[i + 1] = u1[i] + u2[i] * h
        u2[i + 1] = u2[i] + (-p) * h
    return u1, u2

def IVP_Implicit(p, h):
    iteration = round(1/h) + 1
    u1 = np.zeros(iteration)
    u2 = np.zeros(iteration)
    u1[0] = 0
    u2[0] = initial_value(p)
    for i in range(iteration - 1):
        u2[i + 1] = u2[i] + (-p) * h
        u1[i + 1] = u1[i] + u2[i + 1] * h
    return u1, u2

# Additional functions to calculate initial conditions for IVP
def shooting_method(u2_initial_guess, p):
    h = 0.01
    iteration = round(1/h)
    u0 = [0, u2_initial_guess]
    u1 = np.zeros(iteration)
    u2 = np.zeros(iteration)
    u1[0] = u0[0]
    u2[0] = u0[1]
    for i in range(iteration - 1):
        u1[i + 1] = u1[i] + u2[i] * h
        u2[i + 1] = u2[i] + (-p) * h
    return u1[-1]

def initial_value(p):
    closest_u2 = None
    closest_diff = float('inf')
    x = np.linspace(0,7,100)
    for u2_initial_guess in x:
        final_u1 = shooting_method(u2_initial_guess, p)
        diff = abs(final_u1 - 1)
        if diff < closest_diff:
            closest_diff = diff
            closest_u2 = u2_initial_guess
    return closest_u2

# Streamlit app layout
st.title("Couette-Poiseuille Flow Simulation")
st.write("This app simulates Couette-Poiseuille flow using finite difference methods and Euler methods.")
display_equations()

# Display eigenvalues of the Jacobian matrix in larger, green text
eigenvalues = jacobian_eigenvalues()
st.markdown(
    f"<span style='color:yellow; font-size:24px;'>Eigenvalues of the Jacobian: {eigenvalues}</span>",
    unsafe_allow_html=True
)

# User inputs
P = st.slider("Pressure Gradient (P)", min_value=-2.0, max_value=10.0, step=0.5, value=2.0)
N = st.slider("Number of Grid Points (N)", min_value=10, max_value=200, step=10, value=100)
h = st.slider("Step Size for IVP (h)", min_value=0.001, max_value=0.1, step=0.001, value=0.01)

# Calculate solutions
y_bvp, u_numeric_bvp = solve_couette_poiseuille(P, N)
u_analytic = analytical_solution(y_bvp, P)
u_explicit, _ = IVP_Explicit(P, h)
u_implicit, _ = IVP_Implicit(P, h)

# Create y array for IVP solutions
y_ivp = np.linspace(0, 1, len(u_explicit))

# BVP Solution Plot
fig_bvp = go.Figure()
fig_bvp.add_trace(go.Scatter(x=y_bvp, y=u_numeric_bvp, mode='lines+markers', name='Numerical (BVP)', marker=dict(size=5)))
fig_bvp.add_trace(go.Scatter(x=y_bvp, y=u_analytic, mode='lines', name='Analytical'))
fig_bvp.update_layout(
    title="BVP Solution",
    xaxis_title="y",
    yaxis_title="u",
    hovermode="x unified",
    template="plotly_dark"
)
st.plotly_chart(fig_bvp)

# Explicit Euler Solution Plot
fig_explicit = go.Figure()
fig_explicit.add_trace(go.Scatter(x=y_ivp, y=u_explicit, mode='lines+markers', name='Explicit Euler (IVP)', marker=dict(size=5)))
fig_explicit.add_trace(go.Scatter(x=y_bvp, y=u_analytic, mode='lines', name='Analytical'))
fig_explicit.update_layout(
    title="Explicit Euler Solution (IVP)",
    xaxis_title="y",
    yaxis_title="u",
    hovermode="x unified",
    template="plotly_dark"
)
st.plotly_chart(fig_explicit)

# Implicit Euler Solution Plot
fig_implicit = go.Figure()
fig_implicit.add_trace(go.Scatter(x=y_ivp, y=u_implicit, mode='lines+markers', name='Implicit Euler (IVP)', marker=dict(size=5)))
fig_implicit.add_trace(go.Scatter(x=y_bvp, y=u_analytic, mode='lines', name='Analytical'))
fig_implicit.update_layout(
    title="Implicit Euler Solution (IVP)",
    xaxis_title="y",
    yaxis_title="u",
    hovermode="x unified",
    template="plotly_dark"
)
st.plotly_chart(fig_implicit)

# Display maximum error for BVP solution with larger, green text
max_error_bvp = np.max(np.abs(u_numeric_bvp - u_analytic))
st.markdown(
    f"<span style='color:yellow; font-size:20px;'>Maximum absolute error (BVP vs Analytical): {max_error_bvp:.2e}</span>",
    unsafe_allow_html=True
)
