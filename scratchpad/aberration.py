#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

import relvel3 as rv3


PI = 3.1415926535897932384


def rad2deg(angle_rad):
    return angle_rad * (180.0/PI)


def deg2rad(angle_deg):
    return angle_deg * (PI/180.0)


def pos_angle_rad(angle_rad):
    if type(angle_rad) is np.ndarray:
        for j in range(len(angle_rad)):
            if angle_rad[j] < 0.0:
                angle_rad[j] += PI
    else:
        if angle_rad < 0.0:
            angle_rad += PI
    return angle_rad


def classical_aberration(beta, theta_rad):
    slope = np.sin(theta_rad) / (np.cos(theta_rad) + beta)
    phi_rad = pos_angle_rad(np.arctan(slope))
    return phi_rad


def relativistic_aberration(beta, theta_rad):
    gamma = rv3.gamma_ofbeta(beta)
    slope = np.sin(theta_rad) / (gamma * (np.cos(theta_rad) + beta))
    phi_rad = pos_angle_rad(np.arctan(slope))
    return phi_rad


def make_aberration_plot(kind, beta_val_lst):
    plt.figure()
    theta_deg_values = np.linspace(0, 180, 180)
    theta_rad_values = deg2rad(theta_deg_values)
    for beta in beta_val_lst:
        if kind == 'classical':
            phi_rad_values = classical_aberration(beta, theta_rad_values)
            phi_deg_values = rad2deg(phi_rad_values)
            plt.plot(theta_deg_values, phi_deg_values,
                     label="beta=%.2f" % (beta), marker='o', markevery=18)
        elif kind == 'relativistic':
            phi_rad_values = relativistic_aberration(beta, theta_rad_values)
            phi_deg_values = rad2deg(phi_rad_values)
            plt.plot(theta_deg_values, phi_deg_values,
                     label="beta=%.2f" % (beta), marker='+', markevery=13)
        else:
            print("Unknown kind='%s'" % (kind))
            raise ValueError()

    plt.title("%s aberration" % (kind))
    plt.grid(True)
    plt.xlabel("theta (degrees)")
    plt.ylabel("phi (degrees)")
    plt.xlim(0, 180)
    plt.ylim(0, 180)
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    fname = "%s-aberration.pdf" % (kind)
    plt.savefig(fname, format='pdf')


if __name__ == "__main__":
    make_aberration_plot('classical', [0, 0.1, 0.5, 0.866, 0.99])
    make_aberration_plot('relativistic', [0, 0.1, 0.5, 0.866, 0.99])

    # Quick test of angle_between_vectors_rad
    v1 = np.array([1, 1, 0])
    v2 = np.array([-1, -1, 0])
    print("v1=%s v2=%s angle_between_deg=%.3f"
          "" % (v1, v2, rad2deg(rv3.angle_between_vectors_rad(v1, v2))))
