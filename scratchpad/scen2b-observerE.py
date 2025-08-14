#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import scen2b as scen


for beta_S in [0.5, 0.7, 0.8, 0.866, 0.99]:
    # L, D, Z and all lengths in units of light-sec
    L = 1
    D = 1000
    beta_R = 0
    scen.make_plot1(L, beta_R, beta_S, D)

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
        scen.make_plot2(L, beta_R, D, d_E_A_draw_factor)


# Avoid doing these assignments until the end, to avoid accidentally
# "contaminating" any of the earlier function definitions with reading
# global variables.

# Calculate beta_S such that gamma_S will be 2.0
desired_gamma_S = 2.0
beta_S = np.sqrt(1-1/(desired_gamma_S**2))
D = 3
scen.debug_printing(L, beta_R, beta_S, D)

# Make plots that are drawings of the physical positions of all
# relevant objects at interesting times, all times and distances
# measured in the rest frame.

# t=0 Event 1
# q2 Event 2
# q3 Event 3
# e_{2,E} E receives pulse 2
# e_{2,A} A receives pulse 2

def make_position_plot_of_one_time(ax, L, beta_R, beta_S, D, t, Xmax,
                                   desc_str, verbose=False):
    if verbose:
        print("--------------------")
    gamma_S = scen.gamma_fn(beta_S)
    gamma_R = scen.gamma_fn(beta_R)
    Z = scen.Z_for_pulse_reaching_A_at_same_time_as_E(L, D, beta_S, beta_R)
    q2 = scen.q2_fn(L, gamma_S, beta_S, beta_R)
    q3 = scen.q3_fn(L, gamma_R, beta_S, beta_R)

    #plt.title("L=%.1f D=%.1f beta_S=%.3f t=%.3f" % (L, D, beta_S, t))
    A_x = scen.rod_R_center_x_fn(beta_R, t)
    A_y = D
    A_width = L/20
    A_height = L/20
    if verbose:
        print("beta_R=%.3f gamma_R=%.3f" % (beta_R, gamma_R))
        print("beta_S=%.3f gamma_S=%.3f" % (beta_S, gamma_S))

    B_x = scen.rod_S_center_x_fn(L, gamma_R, gamma_S, beta_S, t)
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

    # Draw rod R
    rect = patches.Rectangle((R_left_x, R_y-(R_height/2)),
                             R_width, R_height, fill=False,
                             edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Draw observer A
    rect = patches.Rectangle((A_x-(A_width/2), A_y-(A_height/2)),
                             A_width, A_height, fill=False,
                             edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Draw rod S
    if verbose:
        print("A_x=%.3f B_x=%.3f" % (A_x, B_x))
        print("S_left_x=%.3f S_width=%.3f" % (S_left_x, S_width))
    rect = patches.Rectangle((S_left_x, L/20), S_width, L/10, fill=False,
                             edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    # Draw observer B
    rect = patches.Rectangle((B_x-(B_width/2), B_y-(B_height/2)),
                             B_width, B_height, fill=False,
                             edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    # Draw observer E
    rect = patches.Rectangle((E_x-(E_width/2), E_y-(E_height/2)),
                             E_width, E_height, fill=False,
                             edgecolor='green', linewidth=2)
    ax.add_patch(rect)

    # Draw wavefront of pulse 2, if it has been emitted
    if draw_pulse_2:
        circle = patches.Circle((pulse_2_center_x, pulse_2_center_y),
                                pulse_2_radius, color='blue', fill=False,
                                linewidth=2)
        ax.add_patch(circle)
    # Draw wavefront of pulse 3, if it has been emitted
    if draw_pulse_3:
        circle = patches.Circle((pulse_3_center_x, pulse_3_center_y),
                                pulse_3_radius, color='red', fill=False,
                                linewidth=2)
        ax.add_patch(circle)
    ax.set_aspect('equal', adjustable='box')
    #ax.set_aspect('equal', adjustable='datalim')

    ax.set_xlim(Z-L, Xmax)
    ax.set_ylim(-1, D+1)
    ax.set_title("t=%.3f %s" % (t, desc_str))


def make_plots_of_interesting_times(L, beta_R, beta_S, D):
    print("--------------------")
    gamma_S = scen.gamma_fn(beta_S)
    gamma_R = scen.gamma_fn(beta_R)
    event_lst = scen.interesting_event_list(L, beta_R, beta_S, D)

    # Calculate the max X coordinate of interest among all plots, to
    # make it common for all of them.  It is the right edge of rod S
    # at the maximum time.
    max_t = event_lst[0]['t']
    for event in event_lst:
        if max_t < event['t']:
            max_t = event['t']
    B_x = scen.rod_S_center_x_fn(L, gamma_R, gamma_S, beta_S, max_t)
    S_right_x = B_x + (L/(2*gamma_S))
    Xmax = S_right_x

    num_plots = len(event_lst)
    nplots_per_row = 2
    nplots_per_column = (num_plots + (nplots_per_row - 1)) // nplots_per_row
    print("num_plots=%d nplots_per_row=%d nplots_per_column=%d"
          "" % (num_plots, nplots_per_row, nplots_per_column))
    #fig, ax = plt.subplots(ncols=nplots_per_row, nrows=nplots_per_column)
    fig, ax = plt.subplots(ncols=nplots_per_row,
                           nrows=nplots_per_column,
                           # units are inches?
                           figsize=(10, 8),
                           sharex='all',
                           sharey='all')

    # A failed attempt to make the subplots have less space between
    # them.
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    #plt.title("L=%.1f D=%.1f beta_S=%.3f" % (L, D, beta_S))

    row = 0
    col = 0
    #time_order = 'left-right-then-down'
    time_order = 'top-down-then-right'
    for event in event_lst:
        t = event['t']
        desc_str = event['title']
        print("t=%.3f row=%d col=%d" % (t, row, col))
        make_position_plot_of_one_time(ax[row][col], L, beta_R, beta_S, D, t,
                                       Xmax, desc_str)
        if time_order == 'left-right-then-down':
            col += 1
            if col == nplots_per_row:
                col = 0
                row += 1
        elif time_order == 'top-down-then-right':
            row += 1
            if row == nplots_per_column:
                row = 0
                col += 1
    fig.tight_layout()

    #plt.grid(True)
    #plt.xlabel("x (light-sec)")
    #plt.ylabel("y (light-sec)")
    #plt.legend()
    #plt.tight_layout()
    #fname = "scen2b-L-%.1f-D-%.1f-beta_S-%.3f-t-%.3f.pdf" % (L, D, beta_S, t)
    fname = "scen2b-L-%.1f-D-%.1f-beta_S-%.3f.pdf" % (L, D, beta_S)
    plt.savefig(fname, format='pdf')

L = 1
D = 3
beta_R = 0.0
# Calculate beta_S such that gamma_S will be 2.0
desired_gamma_S = 2.0
beta_S = np.sqrt(1-1/(desired_gamma_S**2))
make_plots_of_interesting_times(L, beta_R, beta_S, D)
