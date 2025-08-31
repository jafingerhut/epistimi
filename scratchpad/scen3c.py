#! /usr/bin/env python3

import numpy as np
import random
import sys

import relvel3 as rv3


# Rod R is moving with arbitrary 3-d velocity v_R relative to the rest
# frame.  Its center reaches position x_R0 at time t=0.  End 1 is at
# offset -R from its center, and end 2 at offset +R from its center,
# for some 3-d vector R.  R is the length-contracted position of the
# ends with R_u used to represent the position of the ends of R at
# rest, without length contraction.

# Corresponding symbols S, v_S, x_S0, and S are used for rod S.

# Event 1 is when end 1 of R meets end 1 of S (the ends at offsets -R
# and -S).

# Event 2 is when end 2 of R meets end 2 of S (the ends at offsets +R
# and +S).

# In the rest frame, we restrict that Event 1 always occurs first, and
# at time t=0.

# Constraints on the above values:

# By event 1 happening at t=0:
# x_S0 = x_R0 + (S-R)

# By event 2 happening at a time q_2 > 0:
# q_2 (v_S - v_R) = 2(R-S)

# Given v_R, v_S, vectors R, S must be such that R-S is equal to k(v_S
# - v_R) for some real number k > 0.

# Then q_2 = 2k.  Stated another way, given v_R, v_S, S, and k, we
# must have R = S + k(v_S - v_R)

# If we further want event 2 to be spacelike separated from event 1,
# then we must have q_2 < f_2, where f_2 is the time it takes light to
# travel from event 1 to event 2 in the rest frame, where light
# propagates isotropically with one-way speed c.

# f_2 = |2R + 2k v_R| / c = |2S + 2k v_S| / c

# 0 < q_2 = 2k < f_2

# That constraint can be written equivalently as the combination of
# these:

# Conditions (A):
# |R + k v_R| > kc
# |S + k v_S| > kc

# Steps to randomly generate an instance of one of these scenarios:

# (1) generate v_R, v_S randomly, double-checking that they are not
# equal to each other, and with the restriction that |v_R| < c and
# |v_S| < c.

# (2) generate S randomly, e.g. in units of light-seconds.

# (3) generate a real random number k > 0

# (4) Calculate R = S + k(v_S - v_R)

# k > 0 automatically causes q_2 > 0.

# If conditions (A) are false, then we can make them true by
# increasing the magnitude of S, calculate the new resulting
# R=S+k(v_S-v_R), and check conditions (A) again.  Eventually by
# increasing S enough, we should eventually find conditions (A) are
# true.

# We could algebraically derive a scaling factor to multiply S by that
# would make conditions (A) true, but I suspect that computationally
# if we simply double S repeatedly until conditions (A) are true, it
# should not take very many doublings.


def generate_random_scen3_instance(verbose=False):
    v_R_fracofc = np.array([0.0, 0.0, 0.0])
    v_S_fracofc = v_R_fracofc
    while rv3.magnitude(v_S_fracofc - v_R_fracofc) < 0.0001:
        v_R_msec = np.array([random.random(), random.random(), random.random()]) * rv3.c
        v_R_fracofc = rv3.random_rescale_if_faster_than_c(v_R_msec, verbose) / rv3.c
        v_S_msec = np.array([random.random(), random.random(), random.random()]) * rv3.c
        v_S_fracofc = rv3.random_rescale_if_faster_than_c(v_S_msec, verbose) / rv3.c
    v_R_msec = v_R_fracofc * rv3.c
    v_S_msec = v_S_fracofc * rv3.c
    # v_R, v_S are now in magnitude range [0,1), i.e. in units of
    # light-seconds per second.
    S = np.array([0.0, 0.0, 0.0])
    while rv3.magnitude(S) <= 0.0:
        S = np.array([random.random(), random.random(), random.random()])
    k = 0.0
    while k <= 0.0:
        k = random.random()
    k_times_velocity_diff = k * (v_S_fracofc - v_R_fracofc)
    R = S + k_times_velocity_diff
    doublings_required = 0
    while True:
        mag = rv3.magnitude(R + k * v_R_fracofc)
        if verbose:
            print("S=%s R=%s k=%s v_R_fracofc=%s magnitude(R+k*v_R)=%s"
                  "" % (S, R, k, v_R_fracofc, mag))
        if mag > k:
            break
        doublings_required += 1
        S = 2 * S
        R = S + k_times_velocity_diff
    ret = {'v_R_fracofc': v_R_fracofc,
           'v_S_fracofc': v_S_fracofc,
           'R_lightsec': R,
           'S_lightsec': S,
           'k': k,
           'doublings_required': doublings_required}
    return ret


# Functions that call check_constraints are only written to return
# correct answers when this function returns True.
def check_constrants(v_R_msec, v_S_msec, R_lightsec, S_lightsec):
    diff_R_S = R_lightsec - S_lightsec
    diff_v_S_v_R = v_S_msec - v_R_msec
    angle_rad = rv3.angle_between_vectors_rad(diff_R_S, diff_v_S_v_R)
    if angle_rad < 0.0001:
        return True
    return False


def calc_q_2(v_R_msec, v_S_msec, R_lightsec, S_lightsec, k):
    assert check_constraints(v_R_msec, v_S_msec, R_lightsec, S_lightsec)
    assert k > 0
    return 2*k


# d_A = e_{2,A} - e_{1,A}
def calc_d_A(v_R_msec, v_S_msec, R_lightsec, S_lightsec, q_2_sec):
    assert check_constraints(v_R_msec, v_S_msec, R_lightsec, S_lightsec)
    gamma_R = rv3.gamma_msec(v_R_msec)
    d_A = q_2 - ((gamma_R**2) * 2 * np.dot(R_lightsec, v_R_msec))
    return d_A


# d_B = e_{2,B} - e_{1,B}
def calc_d_A(v_R_msec, v_S_msec, R_lightsec, S_lightsec, q_2_sec):
    assert check_constraints(v_R_msec, v_S_msec, R_lightsec, S_lightsec)
    gamma_S = rv3.gamma_msec(v_S_msec)
    d_B = q_2 - ((gamma_S**2) * 2 * np.dot(S_lightsec, v_S_msec))
    return d_B


def main():
    for i in range(300):
        x = generate_random_scen3_instance(verbose=False)
        #print("i=%d x=%s" % (i, x))
        if x['doublings_required'] > 0:
            print("Got one!  i=%d x=%s" % (i, x))
            sys.exit(0)


if __name__ == "__main__":
    main()
