#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim

def rod_R_center_x_fn(beta_R, t):
    return 0 + beta_R * t

def rod_S_center_x_fn(L, gamma_R, gamma_S, beta_S, t):
    x_S0 = (-L/2) * ((1/gamma_R) + (1/gamma_S))
    return x_S0 + beta_S * t

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


def make_plot2(L, beta_R, D, d_E_A_draw_factor):
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

    # I multiply this by a factor, because for large values of D, the
    # values of d_E/d_A are otherwise so small as to be
    # indistinguishable from 0.
    E_diff_over_A_diff_values = d_E_A_draw_factor * ((d_E_values / d_A_values) - 1.0)
    plt.plot(beta_S_values, E_diff_over_A_diff_values,
             label='%.0f*((d_E_restclock/d_A_restclock)-1)' % (d_E_A_draw_factor),
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
    beta_R = 0
    plot_lst = []
    plot_lst.append({'D': 3, 'draw_factor': 1})
    plot_lst.append({'D': 1000, 'draw_factor': 1000})
    for plot_info in plot_lst:
        D = plot_info['D']
        d_E_A_draw_factor = plot_info['draw_factor']
        make_plot2(L, beta_R, D, d_E_A_draw_factor)


# Avoid doing these assignments until the end, to avoid accidentally
# "contaminating" any of the earlier function definitions with reading
# global variables.

def debug_printing(L, beta_R, beta_S, D):
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)

    beta_half = beta_half_fn(beta_S, gamma_S, beta_R, gamma_R)
    A_time_diff_Aclock = A_time_diff_Aclock_fn(L, gamma_R, beta_half)
    B_time_diff_Bclock = B_time_diff_Bclock_fn(L, gamma_S, beta_half)

    x_S0 = (-L/2) * ((1/gamma_R) + (1/gamma_S))
    e2B = e2B_fn(L, beta_S, beta_R, D)
    e3B = e3B_fn(L, beta_S, beta_R, D)
    e2A = e2A_fn(L, beta_S, beta_R, D)
    e3A = e3A_fn(L, beta_S, beta_R, D)

    Z = Z_for_pulse_reaching_A_at_same_time_as_E(L, D, beta_S, beta_R)

    print("L=%.1f" % (L))
    print("D=%.1f" % (D))
    print("beta_S=%.3f" % (beta_S))
    print("gamma_S=%.3f" % (gamma_S))
    print("beta_R=%.3f" % (beta_R))
    print("gamma_R=%.3f" % (gamma_R))
    print("Z=%.3f" % (Z))
    print("Z+(beta_S*D)=%.3f" % (Z + (beta_S*D)))
    d_E_restclock = (e3E_fn(L, beta_S, beta_R, D, Z) -
                     e2E_fn(L, beta_S, beta_R, D, Z))
    d_E_Bclock = d_E_restclock / gamma_S
    print("d_E_restclock=%.3f" % (d_E_restclock))
    print("d_E_Bclock=%.3f" % (d_E_Bclock))
    print("q2 (rest clock)=%.3f" % (q2_fn(L, gamma_S, beta_S, beta_R)))
    print("q3 (rest clock)=%.3f" % (q3_fn(L, gamma_R, beta_S, beta_R)))
    print("e2A (rest clock)=%.3f" % (e2A))
    print("e3A (rest clock)=%.3f" % (e3A))
    print("e2E (rest clock)=%.3f" % (e2E_fn(L, beta_S, beta_R, D, Z)))
    print("e3E (rest clock)=%.3f" % (e3E_fn(L, beta_S, beta_R, D, Z)))
    print("beta_half=%.3f" % (beta_half))
    print("A_time_diff (A clock)=%.3f" % (A_time_diff_Aclock))
    print("e3B (rest clock)=%.3f" % (e3B))
    print("e2B (rest clock)=%.3f" % (e2B))
    print("e3B - e2B (rest clock)=%.3f" % (e3B - e2B))
    print("x coord of B at time e3B=%.3f" % (x_S0 + beta_S * e3B))
    print("x coord of B at time e2B=%.3f" % (x_S0 + beta_S * e2B))
    print("B_time_diff (B clock)=%.3f" % (B_time_diff_Bclock))
    #print("Question: Is there a value of Z such that F/gamma_S = 2*(A_time_diff_Aclock)=%.3f ?" % (2*(A_time_diff_Aclock / gamma_R)))
    #print("James question: What value of Z leads to e_{3,E} = e_{3,A}?  Is there always exactly one such value of Z for any given value of the other parameters?  Sometimes no solution?  Sometimes more than one?  What position is E at that time, in A's frame?")
    #print("James question (maybe same answer as previous question): What value of Z leads to E being at A's position at e_{3,A} when A receives the event 3 pulse?  Is there always exactly one such value of Z for any given value of the other parameters?  Sometimes no solution?  Sometimes more than one?")

# Calculate beta_S such that gamma_S will be 2.0
desired_gamma_S = 2.0
beta_S = np.sqrt(1-1/(desired_gamma_S**2))
D = 3
debug_printing(L, beta_R, beta_S, D)

# Make plots that are drawings of the physical positions of all
# relevant objects at interesting times, all times and distances
# measured in the rest frame.

# t=0 Event 1
# q2 Event 2
# q3 Event 3
# e_{2,E} E receives pulse 2
# e_{2,A} A receives pulse 2

def make_position_plot_of_one_time(L, beta_R, beta_S, D, t, Xmax,
                                   verbose=False):
    if verbose:
        print("--------------------")
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)
    Z = Z_for_pulse_reaching_A_at_same_time_as_E(L, D, beta_S, beta_R)
    q2 = q2_fn(L, gamma_S, beta_S, beta_R)
    q3 = q3_fn(L, gamma_R, beta_S, beta_R)

    #plt.title("L=%.1f D=%.1f beta_S=%.3f t=%.3f" % (L, D, beta_S, t))
    A_x = rod_R_center_x_fn(beta_R, t)
    A_y = D
    A_width = L/20
    A_height = L/20
    if verbose:
        print("beta_R=%.3f gamma_R=%.3f" % (beta_R, gamma_R))
        print("beta_S=%.3f gamma_S=%.3f" % (beta_S, gamma_S))

    B_x = rod_S_center_x_fn(L, gamma_R, gamma_S, beta_S, t)
    B_y = D
    B_width = L/20
    B_height = L/20

    E_x = B_x + Z
    E_y = D
    E_width = L/20
    E_height = L/20

    R_left_x = A_x - (L/2)
    R_right_x = A_x + (L/2)
    R_width = L
    R_height = L/10
    R_y = 0
    S_left_x = B_x - (L/(2*gamma_S))
    #S_right_x = B_x + (L/(2*gamma_S))
    S_width = (L/gamma_S)
    # Draw S slightly above R
    S_y = L/10
    S_height = L/10
    pulse_2_center_x = R_left_x
    pulse_2_center_y = 0
    draw_pulse_2 = False
    if verbose:
        print("t=%.3f q2=%.3f q3=%.3f" % (t, q2, q3))
    if t >= q2:
        draw_pulse_2 = True
        pulse_2_radius = (t - q2)
    pulse_3_center_x = R_right_x
    pulse_3_center_y = 0
    draw_pulse_3 = False
    if t >= q3:
        draw_pulse_3 = True
        pulse_3_radius = (t - q3)

    # Update rod R
    rect = global_artists_dict['rod_R']
    rect.set_x(R_left_x)
    rect.set_y(R_y-(R_height/2))
    rect.set_width(R_width)
    rect.set_height(R_height)

    # Update observer A
    rect = global_artists_dict['observer_A']
    rect.set_x(A_x-(A_width/2))
    rect.set_y(A_y-(A_height/2))
    rect.set_width(A_width)
    rect.set_height(A_height)

    # Update rod S
    if verbose:
        print("A_x=%.3f B_x=%.3f" % (A_x, B_x))
        print("S_left_x=%.3f S_width=%.3f" % (S_left_x, S_width))
    rect = global_artists_dict['rod_S']
    rect.set_x(S_left_x)
    rect.set_y(S_y-(S_height/2))
    rect.set_width(S_width)
    rect.set_height(S_height)

    # Update observer B
    rect = global_artists_dict['observer_B']
    rect.set_x(B_x-(B_width/2))
    rect.set_y(B_y-(B_height/2))
    rect.set_width(B_width)
    rect.set_height(B_height)

    # Update observer E
    rect = global_artists_dict['observer_E']
    rect.set_x(E_x-(E_width/2))
    rect.set_y(E_y-(E_height/2))
    rect.set_width(E_width)
    rect.set_height(E_height)

    # Update wavefront of pulse 2
    circle = global_artists_dict['pulse_2_wavefront']
    if draw_pulse_2:
        tmp_radius = pulse_2_radius
    else:
        tmp_radius = 0
    circle.set_radius(tmp_radius)

    # Update wavefront of pulse 3
    circle = global_artists_dict['pulse_3_wavefront']
    if draw_pulse_3:
        tmp_radius = pulse_3_radius
    else:
        tmp_radius = 0
    circle.set_radius(tmp_radius)



global_artists_dict = {}
global_artists_lst = []
global_ax = None
global_L = None
global_beta_R = None
global_beta_S = None
global_D = None
global_Xmax = None
global_min_t = None

def init_globals(L, beta_R, beta_S, D, Xmax, min_t):
    global global_L
    global global_beta_R
    global global_beta_S
    global global_D
    global global_Xmax
    global global_min_t
    global_L = L
    global_beta_R = beta_R
    global_beta_S = beta_S
    global_D = D
    global_Xmax = Xmax
    global_min_t = min_t

def init_animation():
    global global_ax
    global global_L
    global global_beta_R
    global global_beta_S
    global global_D
    global global_Xmax
    global global_artists_dict
    global global_artists_lst

    rect = patches.Rectangle((0, 0), 0, 0, fill=False,
                             edgecolor='black', linewidth=2)
    global_artists_dict['rod_R'] = rect
    global_ax.add_patch(rect)
    global_artists_lst.append(rect)

    rect = patches.Rectangle((0, 0), 0, 0, fill=False,
                             edgecolor='black', linewidth=2)
    global_artists_dict['observer_A'] = rect
    global_ax.add_patch(rect)
    global_artists_lst.append(rect)

    rect = patches.Rectangle((0, 0), 0, 0, fill=False,
                             edgecolor='red', linewidth=2)
    global_artists_dict['rod_S'] = rect
    global_ax.add_patch(rect)
    global_artists_lst.append(rect)

    rect = patches.Rectangle((0, 0), 0, 0, fill=False,
                             edgecolor='red', linewidth=2)
    global_artists_dict['observer_B'] = rect
    global_ax.add_patch(rect)
    global_artists_lst.append(rect)

    rect = patches.Rectangle((0, 0), 0, 0, fill=False,
                             edgecolor='green', linewidth=2)
    global_artists_dict['observer_E'] = rect
    global_ax.add_patch(rect)
    global_artists_lst.append(rect)

    beta_R = global_beta_R
    t = global_min_t
    L = global_L

    A_x = rod_R_center_x_fn(beta_R, t)
    R_left_x = A_x - (L/2)
    R_right_x = A_x + (L/2)
    pulse_2_center_x = R_left_x
    pulse_2_center_y = 0
    pulse_2_radius = 0

    pulse_3_center_x = R_right_x
    pulse_3_center_y = 0
    pulse_3_radius = 0

    circle = patches.Circle((pulse_2_center_x, pulse_2_center_y),
                            pulse_2_radius, color='blue', fill=False,
                            linewidth=2)
    global_artists_dict['pulse_2_wavefront'] = circle
    global_ax.add_patch(circle)
    global_artists_lst.append(circle)

    circle = patches.Circle((pulse_3_center_x, pulse_3_center_y),
                            pulse_3_radius, color='red', fill=False,
                            linewidth=2)
    global_artists_dict['pulse_3_wavefront'] = circle
    global_ax.add_patch(circle)
    global_artists_lst.append(circle)

    make_position_plot_of_one_time(global_L, global_beta_R,
                                   global_beta_S, global_D,
                                   global_min_t, global_Xmax)
    return global_artists_lst


def animate(t):
    global global_L
    global global_beta_R
    global global_beta_S
    global global_D
    global global_Xmax
    return make_position_plot_of_one_time(global_L, global_beta_R,
                                          global_beta_S, global_D,
                                          t, global_Xmax)
    return global_artists_lst


def make_plots_of_interesting_times(L, beta_R, beta_S, D):
    print("--------------------")
    gamma_S = gamma_fn(beta_S)
    gamma_R = gamma_fn(beta_R)
    Z = Z_for_pulse_reaching_A_at_same_time_as_E(L, D, beta_S, beta_R)
    e2E = e2E_fn(L, beta_S, beta_R, D, Z)
    e3E = e3E_fn(L, beta_S, beta_R, D, Z)
    e2A = e2A_fn(L, beta_S, beta_R, D)
    e3A = e3A_fn(L, beta_S, beta_R, D)
    e2B = e2B_fn(L, beta_S, beta_R, D)
    e3B = e3B_fn(L, beta_S, beta_R, D)

    q2 = q2_fn(L, gamma_S, beta_S, beta_R)
    q3 = q3_fn(L, gamma_R, beta_S, beta_R)

    event_lst = []
    event_lst.append({'t': 0, 'title': 'Event 1 - Rods first meet'})
    event_lst.append({'t': q2, 'title': 'Event 2 - Left end of rods meet'})
    event_lst.append({'t': q3, 'title': 'Event 3 - Right end of rods meet'})
    event_lst.append({'t': e2E, 'title': 'Pulse from event 2 reaches E'})
    event_lst.append({'t': e2A, 'title': 'Pulse from event 2 reaches A'})
    event_lst.append({'t': e3A, 'title': 'Pulse from event 3 reaches A and E'})
    event_lst.append({'t': e3B, 'title': 'Pulse from event 3 reaches B'})
    event_lst.append({'t': e2B, 'title': 'Pulse from event 2 reaches B'})

    start_padding = e2B/20
    end_padding = e2B/20
    min_t = 0 - start_padding
    max_t = e2B + end_padding

    # Calculate the max X coordinate of interest among all plots, to
    # make it common for all of them.  It is the right edge of rod S
    # at the maximum time.
    max_t = event_lst[0]['t']
    for event in event_lst:
        if max_t < event['t']:
            max_t = event['t']
    max_t += end_padding

    B_x = rod_S_center_x_fn(L, gamma_R, gamma_S, beta_S, max_t)
    S_right_x = B_x + (L/(2*gamma_S))
    Xmax = S_right_x

    fig, ax = plt.subplots()
    global global_ax
    global_ax = ax
    ax.set_xlim(Z-L, Xmax)
    ax.set_ylim(-1, D+1)
    ax.set_aspect('equal', adjustable='box')

    num_frames = 200
    frame_lst = []
    t = min_t
    delta_t = (max_t - min_t) / (num_frames - 1)
    while len(frame_lst) < num_frames:
        frame_lst.append(t)
        t += delta_t

    init_globals(L, beta_R, beta_S, D, Xmax, min_t)

    # Create the animation
    ani = anim.FuncAnimation(fig, animate, frames=frame_lst, interval=50,
                             blit=False, init_func=init_animation)

    # Create a PillowWriter instance
    writer = anim.PillowWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    # Save the animation using the PillowWriter
    fname = "scen2b-L-%.1f-D-%.1f-beta_S-%.3f.gif" % (L, D, beta_S)
    ani.save(fname, writer=writer)

L = 1
D = 3
beta_R = 0.0
# Calculate beta_S such that gamma_S will be 2.0
desired_gamma_S = 2.0
beta_S = np.sqrt(1-1/(desired_gamma_S**2))
make_plots_of_interesting_times(L, beta_R, beta_S, D)
