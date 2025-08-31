#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import relvel3 as rv3
import scen2b as scen
import aberration as abr


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
    gamma_S = rv3.gamma_ofbeta(beta_S)
    gamma_R = rv3.gamma_ofbeta(beta_R)
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

print("--------------------------------------------------")
print("Calculations of directions, including aberration")
print("--------------------------------------------------")
event_2_source = np.array([-L/2, 0])
event_3_source = np.array([L/2, 0])

gamma_R = rv3.gamma_ofbeta(beta_R)
gamma_S = rv3.gamma_ofbeta(beta_S)

plus_x_direction = np.array([1, 0])

# First calculate directions from which A will receive event 2 and 3
# pulses.  Here the source and receiver are all at rest relative to
# one another, so aberration is 0.

A_location = np.array([0,D])
direction_A_to_ev2 = event_2_source - A_location
direction_A_to_ev3 = event_3_source - A_location
print("direction_A_to_ev2=%s angle with +x axis (deg)=%.3f"
      "" % (direction_A_to_ev2,
            abr.rad2deg(rv3.angle_between_vectors_rad(plus_x_direction,
                                                      direction_A_to_ev2))))
print("direction_A_to_ev3=%s angle with +x axis (deg)=%.3f"
      "" % (direction_A_to_ev3,
            abr.rad2deg(rv3.angle_between_vectors_rad(plus_x_direction,
                                                      direction_A_to_ev3))))

# Consider light pulse emitted at Event 2 from left end of R.
# It reaches B at time e_{2,B} (in A's frame).
# Calculate direction of the pulse that arrives at B at that time.

print("")
e2B = scen.e2B_fn(L, beta_S, beta_R, D)
B_receiving_ev2_pulse = np.array([scen.rod_S_center_x_fn(L, gamma_R, gamma_S, beta_S, e2B), D])
direction_B_to_ev2 = event_2_source - B_receiving_ev2_pulse
print("direction_B_to_ev2=%s" % (direction_B_to_ev2))
direction_B_to_ev2 = scen.normalize_to_unit_vector(direction_B_to_ev2)
print("direction_B_to_ev2 (unit vec)=%s" % (direction_B_to_ev2))
angle_rad = rv3.angle_between_vectors_rad(plus_x_direction, direction_B_to_ev2)
print("angle between +x axis and direction_B_to_ev2 (deg): %.3f"
      "" % (abr.rad2deg(angle_rad)))

print("")
print("Try using relativistic formula for aberration to calculate direction at which observer at B sees light pulse from event 2 arriving")
adjusted_angle_rad = abr.relativistic_aberration(beta_S, angle_rad)
print("aberration-adjusted angle between +x axis and direction_B_to_ev2 (deg): %.3f"
      "" % (abr.rad2deg(adjusted_angle_rad)))


print("")
e3B = scen.e3B_fn(L, beta_S, beta_R, D)
B_receiving_ev3_pulse = np.array([scen.rod_S_center_x_fn(L, gamma_R, gamma_S, beta_S, e3B), D])
direction_B_to_ev3 = event_3_source - B_receiving_ev3_pulse
print("direction_B_to_ev3=%s" % (direction_B_to_ev3))
direction_B_to_ev3 = scen.normalize_to_unit_vector(direction_B_to_ev3)
print("direction_B_to_ev3 (unit vec)=%s" % (direction_B_to_ev3))
angle_rad = rv3.angle_between_vectors_rad(plus_x_direction, direction_B_to_ev3)
print("angle between +x axis and direction_B_to_ev3 (deg): %.3f"
      "" % (abr.rad2deg(angle_rad)))
adjusted_angle_rad = abr.relativistic_aberration(beta_S, angle_rad)
print("aberration-adjusted angle between +x axis and direction_B_to_ev3 (deg): %.3f"
      "" % (abr.rad2deg(adjusted_angle_rad)))

# Now calculate the direction using from B's frame and aberration of
# light from a moving source (rod R) to the receiver at rest (B).
# Hopefully this direction will match what we found above.

# Note; In B's frame, the source location is also (-L/2, 0), because
# that is where the left end of rod R is when event 2 occurs, and
# while B sees R as length-contracted, it sees S as its full rest
# length.

print("")
event_2_source = np.array([-L/2, 0])
B_receiving_ev2_pulse = np.array([0,D])
dir_before_boost_notunit = B_receiving_ev2_pulse - event_2_source
print("dir_before_boost_notunit=%s" % (dir_before_boost_notunit))
dir_before_boost = scen.normalize_to_unit_vector(dir_before_boost_notunit)
print("dir_before_boost=%s" % (dir_before_boost))

angle_deg = abr.rad2deg(rv3.angle_between_vectors_rad(plus_x_direction,
                                                      -dir_before_boost))
print("angle between +x axis and -dir_before_boost (deg): %.3f"
      "" % (angle_deg))

# parallel and perpendicular (perp) here refer to parallel and perpendicular
# parts of the light direction vector, relative to the velocity vector
# of the source rod R, which is moving to the left in the x direction,
# relative to B at speed v_S.

print("Try boost in -x direction")
dir_after_boost_parallel = (dir_before_boost[0] - beta_S) / (1 - (beta_S*dir_before_boost[0]))
dir_after_boost_perp = dir_before_boost[1] / (gamma_S * (1 - beta_S * dir_before_boost[0]))
dir_after_boost_notunit = np.array([dir_after_boost_parallel, dir_after_boost_perp])
#print("dir_after_boost_notunit=%s" % (dir_after_boost_notunit))
dir_after_boost = scen.normalize_to_unit_vector(dir_after_boost_notunit)
print("dir_after_boost=%s" % (dir_after_boost))

print("Try boost in +x direction")
dir_after_boost_parallel = (dir_before_boost[0] + beta_S) / (1 + (beta_S*dir_before_boost[0]))
dir_after_boost_perp = dir_before_boost[1] / (gamma_S * (1 + beta_S * dir_before_boost[0]))
dir_after_boost_notunit = np.array([dir_after_boost_parallel, dir_after_boost_perp])
#print("dir_after_boost_notunit=%s" % (dir_after_boost_notunit))
dir_after_boost = scen.normalize_to_unit_vector(dir_after_boost_notunit)
print("dir_after_boost=%s" % (dir_after_boost))
