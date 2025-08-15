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


def make_position_plot_of_one_time(ax, L, beta_R, beta_S, D, t, Xmax,
                                   desc_str, verbose=False):
    if verbose:
        print("--------------------")
    d = scen.calc_draw_state(L, beta_R, beta_S, D, t, verbose)

    # Draw rod R
    rect = patches.Rectangle((d['R_left_x'], d['R_y'] - (d['R_height']/2)),
                             d['R_width'], d['R_height'], fill=False,
                             edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Draw observer A
    rect = patches.Rectangle((d['A_x'] - (d['A_width']/2), d['A_y'] - (d['A_height']/2)),
                             d['A_width'], d['A_height'], fill=False,
                             edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Draw rod S
    if verbose:
        print("A_x=%.3f B_x=%.3f" % (d['A_x'], d['B_x']))
        print("S_left_x=%.3f S_width=%.3f" % (d['S_left_x'], d['S_width']))
    rect = patches.Rectangle((d['S_left_x'], d['S_y'] - (d['S_height']/2)),
                             d['S_width'], d['S_height'], fill=False,
                             edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    # Draw observer B
    rect = patches.Rectangle((d['B_x'] - (d['B_width']/2), d['B_y'] - (d['B_height']/2)),
                             d['B_width'], d['B_height'], fill=False,
                             edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    # Draw observer E
    rect = patches.Rectangle((d['E_x'] - (d['E_width']/2), d['E_y'] - (d['E_height']/2)),
                             d['E_width'], d['E_height'], fill=False,
                             edgecolor='green', linewidth=2)
    ax.add_patch(rect)

    # Draw wavefront of pulse 2, if it has been emitted
    if d['draw_pulse_2']:
        circle = patches.Circle((d['pulse_2_center_x'], d['pulse_2_center_y']),
                                d['pulse_2_radius'], color='blue', fill=False,
                                linewidth=2)
        ax.add_patch(circle)
    # Draw wavefront of pulse 3, if it has been emitted
    if d['draw_pulse_3']:
        circle = patches.Circle((d['pulse_3_center_x'], d['pulse_3_center_y']),
                                d['pulse_3_radius'], color='red', fill=False,
                                linewidth=2)
        ax.add_patch(circle)
    ax.set_aspect('equal', adjustable='box')
    #ax.set_aspect('equal', adjustable='datalim')

    Z = scen.Z_for_pulse_reaching_A_at_same_time_as_E(L, D, beta_S, beta_R)
    ax.set_xlim(Z-L, Xmax)
    ax.set_ylim(-1, D+1)
    ax.set_title("t=%.3f %s" % (t, desc_str))


# Make plots that are drawings of the physical positions of all
# relevant objects at interesting times, all times and distances
# measured in the rest frame.
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
