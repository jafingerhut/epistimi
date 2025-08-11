#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def gamma_fn(beta):
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return gamma

def q2_fn(L, gamma_S, beta_S, beta_R):
    q2 = L / (gamma_S * (beta_S - beta_R))
    return q2

def q3_fn(L, gamma_R, beta_S, beta_R):
    q3 = L / (gamma_R * (beta_S - beta_R))
    return q3

# Return a beta value with a rapidity that is half of the difference
# between the rapidity of beta_S and beta_R.
def beta_half_fn(beta_S, gamma_S, beta_R, gamma_R):
    beta_half = (beta_S * gamma_S - beta_R * gamma_R) / (gamma_S + gamma_R)
    return beta_half

# Also called (e_{3,A} - e{2,A}) / gamma_R in my scratchpad doc.
# Note that the return value is measured on A's clock, which is why
# there is a division by gamma_R in the formula above.
def A_time_diff_fn(L, gamma_R, beta_half):
    #A_time_diff = gamma_R * L * beta_half
    A_time_diff = L * beta_half
    return A_time_diff

def eA_radical_expr(half_L_over_gamma, beta_R, D):
    radical_expr = (half_L_over_gamma**2) * (beta_R**2) + (1-(beta_R**2)) * ((half_L_over_gamma**2) + (D**2))
    return radical_expr

# calculate e_{2,A}
def e2A_fn(L, beta_S, beta_R, D):
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)
    half_L_over_gamma = L / (2.0 * gamma_R)
    q_2 = L / (gamma_S * (beta_S - beta_R))
    radical_expr = eA_radical_expr(half_L_over_gamma, beta_R, D)
    e2A = q_2 + (gamma_R**2) * ((half_L_over_gamma * beta_R) + np.sqrt(radical_expr))
    return e2A

# calculate e_{3,A}
def e3A_fn(L, beta_S, beta_R, D):
    gamma_R = gamma_fn(beta_R)
    half_L_over_gamma = L / (2.0 * gamma_R)
    q_3 = L / (gamma_R * (beta_S - beta_R))
    radical_expr = eA_radical_expr(half_L_over_gamma, beta_R, D)
    e3A = q_3 + (gamma_R**2) * ((-half_L_over_gamma * beta_R) + np.sqrt(radical_expr))
    return e3A

def eB_radical_expr(half_L_over_gamma, beta_S, D):
    radical_expr = (half_L_over_gamma**2) * (beta_S**2) + (1-(beta_S**2)) * ((half_L_over_gamma**2) + (D**2))
    return radical_expr

# calculate e_{2,B}
def e2B_fn(L, beta_S, beta_R, D):
    gamma_S = gamma_fn(beta_S)
    half_L_over_gamma = L / (2.0 * gamma_S)
    q_2 = L / (gamma_S * (beta_S - beta_R))
    radical_expr = eB_radical_expr(half_L_over_gamma, beta_S, D)
    e2B = q_2 + (gamma_S**2) * ((half_L_over_gamma * beta_S) + np.sqrt(radical_expr))
    return e2B

def e3B_fn(L, beta_S, beta_R, D):
    gamma_S = gamma_fn(beta_S)
    half_L_over_gamma = L / (2.0 * gamma_S)
    q_3 = L / (gamma_R * (beta_S - beta_R))
    radical_expr = eB_radical_expr(half_L_over_gamma, beta_S, D)
    e3B = q_3 + (gamma_S**2) * ((-half_L_over_gamma * beta_S) + np.sqrt(radical_expr))
    return e3B

def e2E_radical_expr(Z_plus, beta_S, D):
    radical_expr = (Z_plus**2) * (beta_S**2) + (1-(beta_S**2)) * ((Z_plus**2) + (D**2))
    return radical_expr

def e2E_fn(L, beta_S, beta_R, D, Z):
    gamma_S = gamma_fn(beta_S)
    half_L_over_gamma = L / (2.0 * gamma_S)
    Z_plus = Z + half_L_over_gamma
    q_2 = L / (gamma_S * (beta_S - beta_R))
    radical_expr = e2E_radical_expr(Z_plus, beta_S, D)
    e2B = q_2 + (gamma_S**2) * ((Z_plus * beta_S) + np.sqrt(radical_expr))
    return e2B

def e3E_radical_expr(Z_minus, beta_S, D):
    radical_expr = (Z_minus**2) * (beta_S**2) + (1-(beta_S**2)) * ((Z_minus**2) + (D**2))
    return radical_expr

def e3E_fn(L, beta_S, beta_R, D, Z):
    gamma_S = gamma_fn(beta_S)
    half_L_over_gamma = L / (2.0 * gamma_S)
    Z_minus = Z - half_L_over_gamma
    q_3 = L / (gamma_R * (beta_S - beta_R))
    radical_expr = e3E_radical_expr(Z_minus, beta_S, D)
    e3B = q_3 + (gamma_S**2) * ((Z_minus * beta_S) + np.sqrt(radical_expr))
    return e3B
    

# Also called (e_{3,B} - e{2,B}) / gamma_S in my scratchpad doc.
# Note that the return value is measured on B's clock, which is why
# there is a division by gamma_S in the formula above.
def B_time_diff_fn(L, gamma_S, beta_half):
    #B_time_diff = - gamma_S * L * beta_half
    B_time_diff = - L * beta_half
    return B_time_diff

def F_fn(L, beta_S, D, Z, verbose=False):
    gamma_S = gamma_fn(beta_S)
    half_L_over_gamma = L / (2.0 * gamma_S)
    Z_minus = Z - half_L_over_gamma
    Z_plus = Z + half_L_over_gamma
    gamma_S2 = gamma_S**2
    Z_minus_radical = np.sqrt(gamma_S2 + ((D/Z_minus)**2))
    Z_plus_radical = np.sqrt(gamma_S2 + ((D/Z_plus)**2))
    denom = ((np.abs(Z_minus) * Z_minus_radical) +
             (np.abs( Z_plus) * Z_plus_radical))
    F = (L * gamma_S) * (-2 * Z * gamma_S) / denom
    if verbose:
        print("--------------------")
        print("Z=%.3f" % (Z))
        print("gamma_S=%.3f" % (gamma_S))
        print("Z_minus=%.3f" % (Z_minus))
        print("Z_plus=%.3f" % (Z_plus))
        print("Z_minus + Z_plus=%.3f" % (Z_minus + Z_plus))
        print("Z_minus_radical=%.3f" % (Z_minus_radical))
        print("Z_plus_radical=%.3f" % (Z_plus_radical))
        print("2*Z=%.3f" % (2*Z))
        print("denom=%.3f" % (denom))
        print("-2*Z*gamma_S=%.3f" % (-2 * Z * gamma_S))
        print("L*gamma_S=%.3f" % (L * gamma_S))
        print("F=%.3f" % (F))
    return F

# Also called (e_{3,E}-e_{2,E}) / gamma_S in my scratchpad doc.
def E_time_diff_fn(L, beta_S, D, Z, beta_half):
    B_time_diff = B_time_diff_fn(L, gamma_S, beta_half)
    F = F_fn(L, beta_S, D, Z)
    E_time_diff = B_time_diff + (F / gamma_S)
    return E_time_diff

# Return value of Z such that observer E will reach A's position
# exactly when A receives the light pulse for Event 3 (and thus E will
# also receive the light pulse for Event 3 at that same time).
def Z_for_pulse_reaching_A_at_same_time_as_E(L, D, beta_S, beta_R):
    # First find time e_{3,A} that pulse reaches A
    e3A = e3A_fn(L, beta_S, beta_R, D)
    # Now find Z value such that E reaches A's position at time e3A.
    # Equation for position of E at A's time t is:
    # (x_{R,0} - (L/2)((1/gamma_R)+(1/gamma_S)) + Z + v_S t, D)
    # Equation for position of A at A's time t is:
    # (x_{R,0} + v_R t, D)
    #
    # Those positions are equal at time e3A if:
    # Z = (L/2)((1/gamma_R)+(1/gamma_S)) + (v_R - v_S) * e3A
    gamma_R = gamma_fn(beta_R)
    gamma_S = gamma_fn(beta_S)
    Z = (L/2)*((1/gamma_R) + (1/gamma_S)) + (beta_R - beta_S) * e3A
    return Z


# L, D, Z and all lengths in units of light-sec
L = 1
D = 1000
#D = 3
beta_S = 0.5
#beta_S = 0.7
#beta_S = 0.8
#beta_S = 0.866
beta_R = 0

gamma_S = gamma_fn(beta_S)
gamma_R = gamma_fn(beta_R)

beta_half = beta_half_fn(beta_S, gamma_S, beta_R, gamma_R)
A_time_diff = A_time_diff_fn(L, gamma_R, beta_half)
B_time_diff = B_time_diff_fn(L, gamma_S, beta_half)

def specialized_F_over_gamma_S_fn(Z, verbose=False):
    return F_fn(L, beta_S, D, Z, verbose) / gamma_S

def specialized_E_time_diff_fn(Z):
    return E_time_diff_fn(L, beta_S, D, Z, beta_half)

x_S0 = (-L/2) * ((1/gamma_R) - (1/gamma_S))
e2B = e2B_fn(L, beta_S, beta_R, D)
e3B = e3B_fn(L, beta_S, beta_R, D)

print("L=%.1f" % (L))
print("D=%.1f" % (D))
print("beta_S=%.3f" % (beta_S))
print("gamma_S=%.3f" % (gamma_S))
print("beta_R=%.3f" % (beta_R))
print("gamma_R=%.3f" % (gamma_R))
print("beta_half=%.3f" % (beta_half))
print("A_time_diff (on A's clock)=%.3f" % (A_time_diff))
print("e3B (in rest frame clock)=%.3f" % (e3B))
print("e2B (in rest frame clock)=%.3f" % (e2B))
print("e3B - e2B (in rest frame clock)=%.3f" % (e3B - e2B))
print("x coord of B at time e3B=%.3f" % (x_S0 + beta_S * e3B))
print("x coord of B at time e2B=%.3f" % (x_S0 + beta_S * e2B))
print("B_time_diff (on B's clock)=%.3f" % (B_time_diff))
print("Question: Is there a value of Z such that F/gamma_S = 2*(A_time_diff / gamma_R)=%.3f ?" % (2*(A_time_diff / gamma_R)))

print("James question: What value of Z leads to e_{3,E} = e_{3,A}?  Is there always exactly one such value of Z for any given value of the other parameters?  Sometimes no solution?  Sometimes more than one?  What position is E at that time, in A's frame?")

print("James question (maybe same answer as previous question): What value of Z leads to E being at A's position at e_{3,A} when A receives the event 3 pulse?  Is there always exactly one such value of Z for any given value of the other parameters?  Sometimes no solution?  Sometimes more than one?")

##for x in [-2*D, 0, 2*D]:
#for x in [-100*D, -2*D]:
#    tmp = specialized_F_over_gamma_S_fn(x, verbose=True)


def specialized_A_time_diff_fn(Z):
    val = A_time_diff_fn(L, gamma_R, beta_half)
    if type(Z) is np.ndarray:
        ret = np.full(len(Z), val)
    else:
        ret = val
    return ret

def specialized_B_time_diff_fn(Z):
    val = B_time_diff_fn(L, gamma_S, beta_half)
    if type(Z) is np.ndarray:
        ret = np.full(len(Z), val)
    else:
        ret = val
    return ret

def specialized_e2E_fn(Z):
    return e2E_fn(L, beta_S, beta_R, D, Z)

def specialized_e3E_fn(Z):
    return e3E_fn(L, beta_S, beta_R, D, Z)


make_plot1 = True
#make_plot1 = False
if make_plot1:
    #x_values = np.linspace(-100*D, 100*D, 200) # 200 points between the limits
    #x_values = np.linspace(-3*D, 3*D, 100) # 100 points between the limits
    x_values = np.linspace(-2*D, 2*D, 100) # 100 points between the limits

    #y_values = specialized_F_over_gamma_S_fn(x_values)
    #plt.plot(x_values, y_values, label='F / gamma_S')

    y_values = specialized_E_time_diff_fn(x_values)
    plt.plot(x_values, y_values, label='E_time_diff / gamma_S')

    y_values2 = specialized_A_time_diff_fn(x_values)
    plt.plot(x_values, y_values2, label='A_time_diff / gamma_R')

    #y_values2x2 = 2 * specialized_A_time_diff_fn(x_values)
    #plt.plot(x_values, y_values2x2, label='2*(A_time_diff / gamma_R)')

    y_values3 = specialized_B_time_diff_fn(x_values)
    plt.plot(x_values, y_values3, label='B_time_diff / gamma_S')

    #e2E_values = specialized_e2E_fn(x_values)
    #plt.plot(x_values, e2E_values, label='e_{2,E}')

    #e3E_values = specialized_e3E_fn(x_values)
    #plt.plot(x_values, e3E_values, label='e_{3,E}')

    plt.title("L=%.1f beta_R=%.3f beta_S=%.3f" % (L, beta_R, beta_S))
    plt.xlabel("Z (light-sec)")
    plt.ylabel("t (sec)")
    plt.legend()

    plt.show()


def specialized_Z_for_pulse_reaching_A_at_same_time_as_E(my_beta_S):
    return Z_for_pulse_reaching_A_at_same_time_as_E(L, D, my_beta_S, beta_R)

def specialized_e3E_minus_e2E_fn(Z):
    return e3E_fn(L, beta_S, beta_R, D, Z) - e2E_fn(L, beta_S, beta_R, D, Z)

def specialized_e3A_minus_e2A_fn(Z):
    val = e3A_fn(L, beta_S, beta_R, D) - e2A_fn(L, beta_S, beta_R, D)
    if type(Z) is np.ndarray:
        ret = np.full(len(Z), val)
    else:
        ret = val
    return ret


make_plot2 = True
if make_plot2:
    x_values = np.linspace(0.001, 0.999, 100) # 100 points between the limits

    Z_values = specialized_Z_for_pulse_reaching_A_at_same_time_as_E(x_values)
    #plt.plot(x_values, Z_values, label='(e_{3,E}-e_{2,E})')

    Z_over_D_values = Z_values / D
    plt.plot(x_values, Z_over_D_values, label='Z/D')

    # I multiply by 100, because otherwise this curve just looks like
    # it is always 0.  It is not always 0, but it takes some
    # "magnification" to see it.
    Z_over_D_plus_beta_values = 100 * ((Z_values / D) + x_values)
    plt.plot(x_values, Z_over_D_plus_beta_values, label='100*((Z/D)+beta)')

    Z_plus_beta_D_values = Z_values + (x_values * D)
    plt.plot(x_values, Z_plus_beta_D_values, label='Z + (beta*D)')

    e3E_minus_e2E_values = specialized_e3E_minus_e2E_fn(Z_values)
    plt.plot(x_values, e3E_minus_e2E_values, label='(e_{3,E}-e_{2,E})')

    e3A_minus_e2A_values = specialized_e3A_minus_e2A_fn(Z_values)
    plt.plot(x_values, e3A_minus_e2A_values, label='(e_{3,A}-e_{2,A})')

    plt.title("L=%.1f D=%.1f beta_R=%.3f beta_S=%.3f" % (L, D, beta_R, beta_S))
    plt.xlabel("beta_S (v_S/c)")
    plt.ylabel("time (sec)")
    plt.legend()
    #plt.show()

    plt.tight_layout()
    fname = "scen2b-where-E-starts-to-receive-pulse-3-at-A-location-D-%.1f-beta_R-%.1f-beta_S-%.1f.pdf" % (D, beta_R, beta_S)
    plt.savefig(fname, format='pdf')
