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
def A_time_diff_Aclock_fn(L, gamma_R, beta_half):
    #A_time_diff_restclock = gamma_R * L * beta_half
    A_time_diff_Aclock = L * beta_half
    return A_time_diff_Aclock

def eA_radical_expr(half_L_over_gamma, beta_R, D):
    radical_expr = (half_L_over_gamma**2) * (beta_R**2) + (1-(beta_R**2)) * ((half_L_over_gamma**2) + (D**2))
    return radical_expr

# calculate e_{2,A} on rest clock
def e2A_fn(L, beta_S, beta_R, D):
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)
    half_L_over_gamma = L / (2.0 * gamma_R)
    q2 = q2_fn(L, gamma_S, beta_S, beta_R)
    radical_expr = eA_radical_expr(half_L_over_gamma, beta_R, D)
    e2A = q2 + (gamma_R**2) * ((half_L_over_gamma * beta_R) + np.sqrt(radical_expr))
    return e2A

# calculate e_{3,A} on rest clock
def e3A_fn(L, beta_S, beta_R, D):
    gamma_R = gamma_fn(beta_R)
    half_L_over_gamma = L / (2.0 * gamma_R)
    q3 = q3_fn(L, gamma_R, beta_S, beta_R)
    radical_expr = eA_radical_expr(half_L_over_gamma, beta_R, D)
    e3A = q3 + (gamma_R**2) * ((-half_L_over_gamma * beta_R) + np.sqrt(radical_expr))
    return e3A

def eB_radical_expr(half_L_over_gamma, beta_S, D):
    radical_expr = (half_L_over_gamma**2) * (beta_S**2) + (1-(beta_S**2)) * ((half_L_over_gamma**2) + (D**2))
    return radical_expr

# calculate e_{2,B} on rest clock
def e2B_fn(L, beta_S, beta_R, D):
    gamma_S = gamma_fn(beta_S)
    half_L_over_gamma = L / (2.0 * gamma_S)
    q2 = q2_fn(L, gamma_S, beta_S, beta_R)
    radical_expr = eB_radical_expr(half_L_over_gamma, beta_S, D)
    e2B = q2 + (gamma_S**2) * ((half_L_over_gamma * beta_S) + np.sqrt(radical_expr))
    return e2B

# calculate e_{3,B} on rest clock
def e3B_fn(L, beta_S, beta_R, D):
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)
    half_L_over_gamma = L / (2.0 * gamma_S)
    q3 = q3_fn(L, gamma_R, beta_S, beta_R)
    radical_expr = eB_radical_expr(half_L_over_gamma, beta_S, D)
    e3B = q3 + (gamma_S**2) * ((-half_L_over_gamma * beta_S) + np.sqrt(radical_expr))
    return e3B

def e2E_radical_expr(Z_plus, beta_S, D):
    radical_expr = (Z_plus**2) * (beta_S**2) + (1-(beta_S**2)) * ((Z_plus**2) + (D**2))
    return radical_expr

# calculate e_{2,E} on rest clock
def e2E_fn(L, beta_S, beta_R, D, Z):
    gamma_S = gamma_fn(beta_S)
    half_L_over_gamma = L / (2.0 * gamma_S)
    Z_plus = Z + half_L_over_gamma
    q2 = q2_fn(L, gamma_S, beta_S, beta_R)
    radical_expr = e2E_radical_expr(Z_plus, beta_S, D)
    e2B = q2 + (gamma_S**2) * ((Z_plus * beta_S) + np.sqrt(radical_expr))
    return e2B

def e3E_radical_expr(Z_minus, beta_S, D):
    radical_expr = (Z_minus**2) * (beta_S**2) + (1-(beta_S**2)) * ((Z_minus**2) + (D**2))
    return radical_expr

# calculate e_{3,E} on rest clock
def e3E_fn(L, beta_S, beta_R, D, Z):
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)
    half_L_over_gamma = L / (2.0 * gamma_S)
    Z_minus = Z - half_L_over_gamma
    q3 = q3_fn(L, gamma_R, beta_S, beta_R)
    radical_expr = e3E_radical_expr(Z_minus, beta_S, D)
    e3B = q3 + (gamma_S**2) * ((Z_minus * beta_S) + np.sqrt(radical_expr))
    return e3B


# Also called (e_{3,B} - e{2,B}) / gamma_S in my scratchpad doc.
# Note that the return value is measured on B's clock, which is why
# there is a division by gamma_S in the formula above.
def B_time_diff_Bclock_fn(L, gamma_S, beta_half):
    #B_time_diff_restclock = - gamma_S * L * beta_half
    B_time_diff_Bclock = - L * beta_half
    return B_time_diff_Bclock

# calculate value of F for Observer E
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
def E_time_diff_Bclock_fn(L, beta_S, gamma_S, D, Z, beta_half):
    B_time_diff_Bclock = B_time_diff_Bclock_fn(L, gamma_S, beta_half)
    F = F_fn(L, beta_S, D, Z)
    E_time_diff_Bclock = B_time_diff_Bclock + (F / gamma_S)
    return E_time_diff_Bclock

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


def make_plot1(L, beta_R, beta_S, D):
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)
    beta_half = beta_half_fn(beta_S, gamma_S, beta_R, gamma_R)

    plt.figure()
    #Z_values = np.linspace(-100*D, 100*D, 200) # 200 points between the limits
    #Z_values = np.linspace(-3*D, 3*D, 100) # 100 points between the limits
    Z_values = np.linspace(-2*D, 2*D, 100) # 100 points between the limits

    E_time_diff_Bclock_values = np.zeros(len(Z_values))
    for j in range(len(Z_values)):
        E_time_diff_Bclock_values[j] = E_time_diff_Bclock_fn(L, beta_S, gamma_S, D, Z_values[j], beta_half)
    plt.plot(Z_values, E_time_diff_Bclock_values, label='E_time_diff (B clock)')

    A_time_diff_Aclock_values = np.zeros(len(Z_values))
    val = A_time_diff_Aclock_fn(L, gamma_R, beta_half)
    for j in range(len(Z_values)):
        A_time_diff_Aclock_values[j] = val
    plt.plot(Z_values, A_time_diff_Aclock_values, label='A_time_diff (A clock)')

    B_time_diff_Bclock_values = np.zeros(len(Z_values))
    val = B_time_diff_Bclock_fn(L, gamma_S, beta_half)
    for j in range(len(Z_values)):
        B_time_diff_Bclock_values[j] = val
    plt.plot(Z_values, B_time_diff_Bclock_values, label='B_time_diff (B clock)')

    plt.title("L=%.1f beta_R=%.3f beta_S=%.3f" % (L, beta_R, beta_S))
    plt.xlabel("Z (light-sec)")
    plt.ylabel("t (sec)")
    plt.legend()

    #plt.show()
    plt.tight_layout()
    fname = "scen2b-A-B-and-E-pulse-time-receive-differences-D-%.1f-beta_R-%.3f-beta_S-%.3f.pdf" % (D, beta_R, beta_S)
    plt.savefig(fname, format='pdf')


def make_plot2(L, beta_R, D):
    plt.figure()
    gamma_R = gamma_fn(beta_R)
    beta_S_values = np.linspace(0.001, 0.999, 100) # 100 points between the limits
    Z_values = Z_for_pulse_reaching_A_at_same_time_as_E(L, D, beta_S_values, beta_R)

    Z_over_D_values = Z_values / D
    plt.plot(beta_S_values, Z_over_D_values, label='Z/D',
             marker='+', markevery=10)

    # I multiply by 100, because otherwise this curve just looks like
    # it is always 0.  It is not always 0, but it takes some
    # "magnification" to see it.
    #Z_over_D_plus_beta_values = 100 * ((Z_values / D) + beta_S_values)
    #plt.plot(beta_S_values, Z_over_D_plus_beta_values, label='100*((Z/D)+beta)')

    Z_plus_beta_D_values = Z_values + (beta_S_values * D)
    plt.plot(beta_S_values, Z_plus_beta_D_values, label='Z + (beta*D)',
             marker='o', markevery=10)

    d_E_values = np.zeros(len(Z_values))
    for j in range(len(Z_values)):
        d_E_values[j] = (e3E_fn(L, beta_S_values[j], beta_R, D, Z_values[j]) -
                         e2E_fn(L, beta_S_values[j], beta_R, D, Z_values[j]))
    plt.plot(beta_S_values, d_E_values, label='d_E_restclock',
             marker='*', markevery=10)

    d_E_Bclock_values = np.zeros(len(Z_values))
    for j in range(len(Z_values)):
        d_E_Bclock_values[j] = d_E_values[j] / gamma_fn(beta_S_values[j])
    plt.plot(beta_S_values, d_E_Bclock_values, label='d_E_Bclock',
             marker='s', markevery=10)

    Aclock_same_as_restclock = False
    if beta_R == 0.0:
        Aclock_same_as_restclock = True
        print("Skipping the plot of separate curve for d_A on A clock, because it is same as the one for rest clock")

    d_A_values = np.zeros(len(Z_values))
    for j in range(len(Z_values)):
        d_A_values[j] = (e3A_fn(L, beta_S_values[j], beta_R, D) -
                         e2A_fn(L, beta_S_values[j], beta_R, D))
    if Aclock_same_as_restclock:
        curve_label = 'd_A_restclock (A and rest clock same)'
    else:
        curve_label = 'd_A_restclock'
    plt.plot(beta_S_values, d_A_values, label=curve_label,
             marker='x', markevery=12)

    if not Aclock_same_as_restclock:
        d_A_Aclock_values = np.zeros(len(Z_values))
        for j in range(len(Z_values)):
            d_A_Aclock_values[j] = d_A_values[j] / gamma_R
        plt.plot(beta_S_values, d_A_Aclock_values, label='d_A_Aclock',
                 marker='-', markevery=10)

    # I multiply this by 1000, because the values are otherwise so
    # small as to be indistinguishable from 0.
    E_diff_over_A_diff_values = 1000.0 * ((d_E_values / d_A_values) - 1.0)
    plt.plot(beta_S_values, E_diff_over_A_diff_values, label='1000*((d_E_restclock/d_A_restclock)-1)',
             marker='v', markevery=10)

    plt.title("L=%.1f D=%.1f beta_R=%.3f" % (L, D, beta_R))
    plt.grid(True)
    plt.xlabel("beta_S (v_S/c)")
    plt.ylabel("time (sec)")
    plt.legend()
    #plt.show()

    plt.tight_layout()
    fname = "scen2b-where-E-starts-to-receive-pulse-3-at-A-location-D-%.1f-beta_R-%.3f.pdf" % (D, beta_R)
    plt.savefig(fname, format='pdf')


for beta_S in [0.5, 0.7, 0.8, 0.866, 0.99]:
    # L, D, Z and all lengths in units of light-sec
    L = 1
    D = 1000
    beta_R = 0
    make_plot1(L, beta_R, beta_S, D)

enable_plot2 = True
if enable_plot2:
    L = 1
    D = 1000
    beta_R = 0
    make_plot2(L, beta_R, D)


# Avoid doing these assignments until the end, to avoid accidentally
# "contaminating" any of the earlier function definitions with reading
# global variables.

def debug_printing(L, beta_R, beta_S, D):
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)

    beta_half = beta_half_fn(beta_S, gamma_S, beta_R, gamma_R)
    A_time_diff_Aclock = A_time_diff_Aclock_fn(L, gamma_R, beta_half)
    B_time_diff_Bclock = B_time_diff_Bclock_fn(L, gamma_S, beta_half)

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
    print("A_time_diff (on A's clock)=%.3f" % (A_time_diff_Aclock))
    print("e3B (in rest frame clock)=%.3f" % (e3B))
    print("e2B (in rest frame clock)=%.3f" % (e2B))
    print("e3B - e2B (in rest frame clock)=%.3f" % (e3B - e2B))
    print("x coord of B at time e3B=%.3f" % (x_S0 + beta_S * e3B))
    print("x coord of B at time e2B=%.3f" % (x_S0 + beta_S * e2B))
    print("B_time_diff (on B's clock)=%.3f" % (B_time_diff_Bclock))
    print("Question: Is there a value of Z such that F/gamma_S = 2*(A_time_diff_Aclock)=%.3f ?" % (2*(A_time_diff_Aclock / gamma_R)))
    print("James question: What value of Z leads to e_{3,E} = e_{3,A}?  Is there always exactly one such value of Z for any given value of the other parameters?  Sometimes no solution?  Sometimes more than one?  What position is E at that time, in A's frame?")
    print("James question (maybe same answer as previous question): What value of Z leads to E being at A's position at e_{3,A} when A receives the event 3 pulse?  Is there always exactly one such value of Z for any given value of the other parameters?  Sometimes no solution?  Sometimes more than one?")

debug_printing(L, beta_R, beta_S, D)
