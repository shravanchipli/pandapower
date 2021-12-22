from numpy import roots, conj, r_, linalg, append
from numpy.core.umath import exp


def _iwamoto_step(Ybus, Sbus, J, F, dx, pq, npv, npq, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4, j5, j6, opt_multipliers, linesearch):
    if npv:
        dVa[pv] = dx[j1:j2]
    if npq:
        dVa[pq] = dx[j3:j4]
        dVm[pq] = dx[j5:j6]
    dV = dVm * exp(1j * dVa)

    if linesearch:
        opt_multiplier = line_search(Va, Vm, dVa, dVm, Ybus, Sbus, pv, pq)
    else:
        opt_multiplier = _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pq, pv)
    print("optimal multiplier:", opt_multiplier)
    opt_multipliers = append(opt_multipliers, opt_multiplier)

    Vm += opt_multiplier * dVm
    Va += opt_multiplier * dVa
    return Vm, Va, opt_multipliers


def _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pq, pv):
    """
    Calculates the iwamato multiplier to increase convergence
    """

    c0=-F                               # c0 = ys-y(x)= -F
    c1=-J * dx                          # c1 = -Jdx
    c2=-_evaluate_Yx(Ybus, dV, pv, pq)  # c2 = -y(dx)

    g0 = c0.dot(c1)
    g1 = c1.dot(c1) + 2 * c0.dot(c2)
    g2 = 3.0 * c1.dot(c2)
    g3 = 2.0 * c2.dot(c2)

    np_roots = roots([g3, g2, g1, g0])[2].real

    return np_roots


def _evaluate_Yx(Ybus, V, pv, pq):
    ## evaluate y(x)
    Yx = V * conj(Ybus * V)
    F = r_[Yx[pv].real,
           Yx[pq].real,
           Yx[pq].imag]
    return F


def line_search(Va, Vm, dVa, dVm, Ybus, Sbus, pv, pq):

    a = 0
    b = 1
    tol = 1.0
    i = 0

    while tol > 10e-8 and i < 200:

        midpoint = a + (b - a)/2

        Vm1 = Vm + a * dVm
        Va1 = Va + a * dVa
        V1  = Vm1 * exp(1j * Va1)

        Vm2 = Vm + midpoint * dVm
        Va2 = Va + midpoint * dVa
        V2  = Vm2 * exp(1j * Va2)

        Vm3 = Vm + b * dVm
        Va3 = Va + b * dVa
        V3  = Vm3 * exp(1j * Va3)

        F1 = _evaluate_Fx(Ybus, V1, Sbus, pv, pq)
        F2 = _evaluate_Fx(Ybus, V2, Sbus, pv, pq)
        F3 = _evaluate_Fx(Ybus, V3, Sbus, pv, pq)

        # cost function
        C1 = 0.5 * linalg.norm(F1)
        C2 = 0.5 * linalg.norm(F2)
        C3 = 0.5 * linalg.norm(F3)

        if C1 > C2:
            a = (midpoint - a)/2
        else:
            if C3 > C1:
                b = midpoint
            else:
                a = midpoint
            continue

        if C3 > C2:
            b = midpoint + (b - midpoint)/2
        else:
            a = midpoint 

        tol = abs(b - a)
        i += 1

    return midpoint
        

def _evaluate_Fx(Ybus, V, Sbus, pv, pq):
    # evalute F(x)
    mis = V * conj(Ybus * V) - Sbus
    F = r_[mis[pv].real,
           mis[pq].real,
           mis[pq].imag]
    return F
        
