""" Electric circuit"""

from SolveLinear import *
from Lab04_Q1_b import *

def build_matrix(R1, R2, R3, R4, R5, R6, x_plus, C1, C2, omega):
    """Builds matrix from Newman Exercise 6.5b"""
    j = 1j
    G1 = 1.0 / R1
    G2 = 1.0 / R2
    G3 = 1.0 / R3
    G4 = 1.0 / R4
    G5 = 1.0 / R5
    G6 = 1.0 / R6

    Yc1 = j * omega * C1
    Yc2 = j * omega * C2

    A = np.zeros((3, 3), dtype=complex)

    A[0, 0] = G1 + G4 + Yc1
    A[0, 1] = -Yc1
    A[0, 2] = 0.0

    A[1, 0] = -Yc1
    A[1, 1] = G2 + G5 + Yc1 + Yc2
    A[1, 2] = -Yc2

    A[2, 0] = 0.0
    A[2, 1] = -Yc2
    A[2, 2] = G3 + G6 + Yc2

    V = np.zeros(3, dtype=complex)
    V[0] = x_plus * G1
    V[1] = x_plus * G2
    V[2] = x_plus * G3

    return A, V

def PartialPivotC(A_in, v_in, m):
    """Perform partial pivoting on column m in-place on copies and return them."""
    A = np.copy(A_in).astype(complex)
    v = np.copy(v_in).astype(complex)
    N = A.shape[0]

    pivot_offset = np.argmax(np.abs(A[m:, m]))
    pivot_row = pivot_offset + m

    if np.isclose(A[pivot_row, m], 0.0):
        raise ValueError(f"Matrix is singular or nearly singular at column {m}")

    if pivot_row != m:
        A[[m, pivot_row], :] = A[[pivot_row, m], :]
        v[m], v[pivot_row] = v[pivot_row], v[m]

    return A, v, pivot_row

def GaussElimPPC(A_in, v_in):
    """Gaussian elimination with partial pivoting; works for complex arrays."""
    A = np.copy(A_in).astype(complex)
    v = np.copy(v_in).astype(complex)
    N = len(v)

    for m in range(N):
        A, v, _ = PartialPivotC(A, v, m)
        div = A[m, m]
        if np.isclose(div, 0.0):
            raise ValueError(f"Zero pivot encountered at row {m}")
        A[m, :] /= div
        v[m] /= div

        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult * A[m, :]
            v[i] -= mult * v[m]

    x = np.empty(N, dtype=complex)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i] * x[i]
    return x

def solve_and_print_plot(R_vals, C1, C2, x_plus, omega, title_suffix="(R6 = resistor)"):
    R1, R2, R3, R4, R5, R6 = R_vals
    A, V = build_matrix(R1, R2, R3, R4, R5, R6, x_plus, C1, C2, omega)
    x = GaussElimPPC(A, V)
    amps = np.abs(x)
    phases_deg = np.angle(x, deg=True)

    print(title_suffix)
    for i in range(3):
        print(f"V{i+1}: amplitude = {amps[i]:.6f} V, phase = {phases_deg[i]:.3f} deg")

    #Time-domain real voltages: v(t) = Re(x * exp(i omega t))
    period = 2*np.pi / omega
    t = np.linspace(0.0, 2*period, 1000)
    expo = np.exp(1j * omega * t)
    vt = np.empty((3, t.size))
    for i in range(3):
        vt[i, :] = np.real(x[i] * expo)

    plt.figure()
    plt.plot(t, vt[0, :], label='V1 (real)')
    plt.plot(t, vt[1, :], label='V2 (real)')
    plt.plot(t, vt[2, :], label='V3 (real)')
    plt.xlabel('time (s)')
    plt.ylabel('voltage (V)')
    plt.title(f'Voltages vs time {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    R1 = R3 = R5 = 1e3 #ohms
    R2 = R4 = 2e3  #ohms
    R6_res = 2e3  #resistor case (ohms)
    C1 = 1e-6  #F
    C2 = 0.5e-6  #F
    x_plus = 3.0  #V
    omega = 1000.0  #rad/s

    # Case 1: original resistor R6
    R_vals_resistor = (R1, R2, R3, R4, R5, R6_res)
    solve_and_print_plot(R_vals_resistor, C1, C2, x_plus, omega, title_suffix="(R6 = 2 kΩ resistor)")

    #Case 2: replace R6 with an inductor
    Z6_inductor = 1j * R6_res
    R_vals_inductor = (R1, R2, R3, R4, R5, Z6_inductor)
    solve_and_print_plot(R_vals_inductor, C1, C2, x_plus, omega,
                         title_suffix="(R6 replaced by inductor L with jωL = j*R6)")



