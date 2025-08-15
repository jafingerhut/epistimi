#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim

import scen2b as scen


def update_artist_objects_for_time_t(L, beta_R, beta_S, D, t, Xmax,
                                     verbose=False):
    if verbose:
        print("--------------------")
    d = scen.calc_draw_state(L, beta_R, beta_S, D, t, verbose)

    # Update rod R
    rect = global_artists_dict['rod_R']
    rect.set_x(d['R_left_x'])
    rect.set_y(d['R_y'] - (d['R_height']/2))
    rect.set_width(d['R_width'])
    rect.set_height(d['R_height'])

    # Update observer A
    rect = global_artists_dict['observer_A']
    rect.set_x(d['A_x'] - (d['A_width']/2))
    rect.set_y(d['A_y'] - (d['A_height']/2))
    rect.set_width(d['A_width'])
    rect.set_height(d['A_height'])

    # Update rod S
    if verbose:
        print("A_x=%.3f B_x=%.3f" % (A_x, B_x))
        print("S_left_x=%.3f S_width=%.3f" % (S_left_x, S_width))
    rect = global_artists_dict['rod_S']
    rect.set_x(d['S_left_x'])
    rect.set_y(d['S_y'] - (d['S_height']/2))
    rect.set_width(d['S_width'])
    rect.set_height(d['S_height'])

    # Update observer B
    rect = global_artists_dict['observer_B']
    rect.set_x(d['B_x'] - (d['B_width']/2))
    rect.set_y(d['B_y'] - (d['B_height']/2))
    rect.set_width(d['B_width'])
    rect.set_height(d['B_height'])

    # Update observer E
    rect = global_artists_dict['observer_E']
    rect.set_x(d['E_x'] - (d['E_width']/2))
    rect.set_y(d['E_y'] - (d['E_height']/2))
    rect.set_width(d['E_width'])
    rect.set_height(d['E_height'])

    # Update wavefront of pulse 2
    circle = global_artists_dict['pulse_2_wavefront']
    if d['draw_pulse_2']:
        tmp_radius = d['pulse_2_radius']
    else:
        tmp_radius = 0
    circle.set_radius(tmp_radius)

    # Update wavefront of pulse 3
    circle = global_artists_dict['pulse_3_wavefront']
    if d['draw_pulse_3']:
        tmp_radius = d['pulse_3_radius']
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

    A_x = scen.rod_R_center_x_fn(beta_R, t)
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

    update_artist_objects_for_time_t(global_L, global_beta_R,
                                     global_beta_S, global_D,
                                     global_min_t, global_Xmax)
    return global_artists_lst


def animate(t):
    global global_L
    global global_beta_R
    global global_beta_S
    global global_D
    global global_Xmax
    update_artist_objects_for_time_t(global_L, global_beta_R,
                                     global_beta_S, global_D,
                                     t, global_Xmax)
    return global_artists_lst


def make_plots_of_interesting_times(L, beta_R, beta_S, D):
    gamma_R = scen.gamma_fn(beta_R)
    gamma_S = scen.gamma_fn(beta_S)
    Z = scen.Z_for_pulse_reaching_A_at_same_time_as_E(L, D, beta_S, beta_R)
    event_lst = scen.interesting_event_list(L, beta_R, beta_S, D)

    # Calculate the max X coordinate of interest among all plots, to
    # make it common for all of them.  It is the right edge of rod S
    # at the maximum time.
    min_t = event_lst[0]['t']
    max_t = event_lst[0]['t']
    for event in event_lst:
        if min_t > event['t']:
            min_t = event['t']
        if max_t < event['t']:
            max_t = event['t']
    duration = max_t - min_t
    start_padding = duration/20
    min_t -= start_padding
    end_padding = duration/20
    max_t += end_padding

    B_x = scen.rod_S_center_x_fn(L, gamma_R, gamma_S, beta_S, max_t)
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
