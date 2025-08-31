#! /usr/bin/env python3

import numpy as np
import random

c=299792458.0


# v_msec must be a vector with magnitude in units of meters/sec
def gamma_msec(v_msec):
    v2 = np.dot(v_msec, v_msec)
    return 1.0 / np.sqrt(1.0 - v2 / c**2)


# v_fracofc must be a vector with magnitude that is a fraction of
# speed of light c
def gamma_fracofc(v_fracofc):
    v2 = np.dot(v_fracofc, v_fracofc)
    return 1.0 / np.sqrt(1.0 - v2)


# beta must be a real number, not a vector, in units of a fraction of
# the speed of light c
def gamma_ofbeta(beta):
    return 1.0 / np.sqrt(1.0 - beta**2)


def magnitude(v):
    return np.sqrt(np.dot(v, v))


def angle_between_vectors_rad(v1, v2):
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    dotp = np.dot(v1, v2)
    angle_rad = np.arccos(dotp/(mag1 * mag2))
    return angle_rad


# Calculate and return vB \oplus u
# This is done according to one of 3 equations, all 3 proven
# algebraically to give the same result, in the Appendix section
# titled "Relativistic velocity addition in 3 dimensions" in my
# scratchpad document.
# vB, u expected to have magnitude in range 0 to c,
# and return value is same.
# NOT relativistic magnitudes where 1 means c.
def add(vB, u):
    normvB2 = np.dot(vB, vB)
    vB_dot_u = np.dot(vB, u)
    gamma_B = gamma_msec(vB)
    factor = 1.0 / (gamma_B * (1.0 + vB_dot_u / (c**2)))
    return factor * (u + (((gamma_B-1) * vB_dot_u / normvB2) + gamma_B) * vB)


# Calculate and return vA \ominus vB
# This is done by calculating the parallel and perpendicular
# comoponents of the relativistic velocity difference, then adding
# those two vectors together.  You can find equations for this in the
# Appendix section titled "Relativistic velocity addition in 3
# dimensions" in my scratchpad document.
# vA, vB expected to have magnitude in range 0 to c,
# and return value is same.
# NOT relativistic magnitudes where 1 means c.
def subtract(vA, vB):
    vA_dot_vB = np.dot(vA, vB)
    normvB2 = np.dot(vB, vB)
    gamma_B = gamma_msec(vB)
    vA_parallel = (vA_dot_vB / normvB2) * vB
    vA_perp = vA - vA_parallel
    denom = 1 - (vA_dot_vB / (c**2))
    u_prime_parallel = (vA_parallel - vB) / denom
    u_prime_perp = (vA_perp / gamma_B) / denom
    return u_prime_parallel + u_prime_perp



def random_rescale_if_faster_than_c(v, verbose=False):
    mag1 = magnitude(v)
    if mag1 < c:
        return v
    new_beta = random.random()
    v = (v / mag1) * c * new_beta
    mag2 = magnitude(v)
    if verbose:
        print(f"|v|={mag1} >= c, rescaling it so its direction is same, but its magnitude is a uniform random value in the range [0,c) mag2={mag2}")
    if mag2 >= c:
        print(f"BUG in program.  New magnitude after rescaling is > c.  v={v} mag2={mag2}")
        sys.exit(1)
    return v


def abs_diff(u, v):
    diff = u - v
    err = np.dot(diff, diff)
    return err


def check_add_commutative(test_case):
    vA = test_case['vA']
    vB = test_case['vB']
    sum1 = add(vA, vB)
    sum2 = add(vB, vA)
    err = abs_diff(sum1, sum2)
    results = ["vA oplus vB=%s" % (sum1/c),
               "vB oplus vA=%s" % (sum2/c)]
    if err > 0.0001:
        ret = {'pass': False,
               'msg': ("vA=%s vB=%s add is NOT commutative, err=%s"
                       "" % (vA/c, vB/c, err)),
               'results': results}
    else:
        ret = {'pass': True,
               'msg': ("vA=%s vB=%s add is commutative, err=%s"
                       "" % (vA/c, vB/c, err)),
               'results': results}
    return ret


def check_subtract_sort_of_commutative(test_case):
    vA = test_case['vA']
    vB = test_case['vB']
    diff1 = subtract(vA, vB)
    diff2 = subtract(vB, vA)
    err = abs_diff(diff1, diff2)
    results = ["vA ominus vB=%s" % (diff1/c),
               "vB ominus vA=%s" % (diff2/c)]
    if err > 0.0001:
        ret = {'pass': False,
               'msg': ("vA=%s vB=%s subtract is NOT just the negative result when swapped, err=%s"
                       "" % (vA/c, vB/c, err)),
               'results': results}
    else:
        ret = {'pass': True,
               'msg': ("vA=%s vB=%s subtract is the negative result when swapped, err=%s"
                       "" % (vA/c, vB/c, err)),
               'results': results}
    return ret


def check_0_minus_a_negates(test_case):
    vA = test_case['vA']
    diff1 = subtract(np.array([0.0, 0.0, 0.0]), vA)
    err = abs_diff(diff1, -vA)
    results = ["0 ominus vA=%s" % (diff1/c)]
    if err > 0.0001:
        ret = {'pass': False,
               'msg': ("0 ominus vA=%s is NOT equal to -vA, err=%s"
                       "" % (vA/c, err)),
               'results': results}
    else:
        ret = {'pass': True,
               'msg': ("0 ominus vA=%s is equal to -vA, err=%s"
                       "" % (vA/c, err)),
               'results': results}
    return ret


def check_a_plus_b_minus_a_left_assoc(test_case):
    vA = test_case['vA']
    vB = test_case['vB']
    val = subtract(add(vA, vB), vA)
    err = abs_diff(val, vB)
    results = ["(vA oplus vB) ominus vA=%s" % (val/c)]
    if err > 0.0001:
        ret = {'pass': False,
               'msg': ("vA=%s vB=%s (vA oplus vB) ominus vA is NOT vB, err=%s"
                       "" % (vA/c, vB/c, err)),
               'results': results}
    else:
        ret = {'pass': True,
               'msg': ("vA=%s vB=%s (vA oplus vB) ominus vA is vB, err=%s"
                       "" % (vA/c, vB/c, err)),
               'results': results}
    return ret


def check_a_plus_b_minus_a_right_assoc(test_case):
    vA = test_case['vA']
    vB = test_case['vB']
    val = add(vA, subtract(vB, vA))
    err = abs_diff(val, vB)
    results = ["vA oplus (vB ominus vA)=%s" % (val/c)]
    if err > 0.0001:
        ret = {'pass': False,
               'msg': ("vA=%s vB=%s vA oplus (vB ominus vA) is NOT vB, err=%s"
                       "" % (vA/c, vB/c, err)),
               'results': results}
    else:
        ret = {'pass': True,
               'msg': ("vA=%s vB=%s vA oplus (vB ominus vA) is vB, err=%s"
                       "" % (vA/c, vB/c, err)),
               'results': results}
    return ret


def check_alternate_method_to_calculate_u_ominus_v(test_case):
    u = test_case['vA']
    v = test_case['vB']

    # Method 1: straightforward way
    diff_vec = subtract(u, v)
    diff_beta_method1 = magnitude(diff_vec) / c

    # Method 2: A way I found with a lot of by-hand algebraic
    # manipulation, that I am suspicious whether I made any mistakes.
    beta_vec_u = u / c
    beta_vec_v = v / c
    beta_u2 = np.dot(beta_vec_u, beta_vec_u)
    beta_v2 = np.dot(beta_vec_v, beta_vec_v)
    theta_rad = angle_between_vectors_rad(u, v)
    sin_theta2 = np.sin(theta_rad)**2
    K = 1.0 - (np.dot(u, v) / (c**2))
    beta_diff_vec = beta_vec_u - beta_vec_v
    beta_diff_mag2 = np.dot(beta_diff_vec, beta_diff_vec)
    F2 = beta_diff_mag2 - (beta_u2 * beta_v2 * sin_theta2)
    diff_beta_method2 = np.sqrt(F2 / (K**2))

    err = abs_diff(diff_beta_method1, diff_beta_method2)
    results = ["diff_beta_method1=%.5f" % (diff_beta_method1),
               "diff_beta_method2=%.5f" % (diff_beta_method2)]
    if err > 0.0001:
        ret = {'pass': False,
               'msg': ("u=%s v=%s alternate method does NOT match |u ominus v|, err=%s"
                       "" % (u/c, v/c, err)),
               'results': results}
    else:
        ret = {'pass': True,
               'msg': ("u=%s v=%s alternate method matches |u ominus v|, err=%s"
                       "" % (u/c, v/c, err)),
               'results': results}
    return ret


def check_u_ominus_v_equals_v_ominus_u(test_case):
    u = test_case['vA']
    v = test_case['vB']

    diff1_vec = subtract(u, v)
    diff1_mag = magnitude(diff1_vec)

    diff2_vec = subtract(v, u)
    diff2_mag = magnitude(diff2_vec)

    err = abs_diff(diff1_mag, diff2_mag)
    results = ["|u ominus v|=%.5f" % (diff1_mag),
               "|v ominus u|=%.5f" % (diff2_mag)]
    if err > 0.0001:
        ret = {'pass': False,
               'msg': ("u=%s v=%s |u ominus v| != |v ominus u|, err=%s"
                       "" % (u/c, v/c, err)),
               'results': results}
    else:
        ret = {'pass': True,
               'msg': ("u=%s v=%s |u ominus v| == |v ominus u|, err=%s"
                       "" % (u/c, v/c, err)),
               'results': results}
    return ret


def check_u_oplus_v_equals_v_oplus_u(test_case):
    u = test_case['vA']
    v = test_case['vB']

    sum1_vec = add(u, v)
    sum1_mag = magnitude(sum1_vec)

    sum2_vec = add(v, u)
    sum2_mag = magnitude(sum2_vec)

    err = abs_diff(sum1_mag, sum2_mag)
    results = ["|u oplus v|=%.5f" % (sum1_mag),
               "|v oplus u|=%.5f" % (sum2_mag)]
    if err > 0.0001:
        ret = {'pass': False,
               'msg': ("u=%s v=%s |u oplus v| != |v oplus u|, err=%s"
                       "" % (u/c, v/c, err)),
               'results': results}
    else:
        ret = {'pass': True,
               'msg': ("u=%s v=%s |u oplus v| == |v oplus u|, err=%s"
                       "" % (u/c, v/c, err)),
               'results': results}
    return ret


def run_test_cases(test_cases, desc_str, check_fn):
    print("")
    good = 0
    for test_case in test_cases:
        ret = check_fn(test_case)
        if ret['pass']:
            good += 1
        print("--------------------")
        print(ret['msg'])
        for s in ret['results']:
            print(s)
    result = "YES"
    if good < len(test_cases):
        result = "NO (%d good out of %d)" % (good, len(test_cases))
    print("----------------------------------------")
    print("Test: %s" % (desc_str))
    print("Result: %s" % (result))


def main():
    test_cases = [
        {'vA': np.array([0.0, -0.3 * c, 0.0]),
         'vB': np.array([0.0,  0.5 * c, 0.0])
         },
        {'vA': np.array([0.1 * c, 0.0, 0.0]),
         'vB': np.array([0.0, 0.1 * c, 0.0])
         },
        {'vA': np.array([0.6 * c, 0.0, 0.0]),
         'vB': np.array([0.5 * c, 0.2 * c, 0.0])
         },
        {'vA': np.array([0.5 * c, 0.0, 0.0]),
         'vB': np.array([0.0, 0.5 * c, 0.0])
         },
        {'vA': np.array([0.4 * c, 0.2 * c, 0.0]),   # B at 0.4c
         'vB': np.array([-0.1 * c, -0.3 * c, 0.0])  # C at -0.1c
         },
        {'vA': np.array([0.9 * c, 0.1 * c, 0.1 * c]),
         'vB': np.array([-0.4 * c, 0.7 * c, 0.2 * c])
         },
    ]
    run_test_cases(test_cases, "a \\oplus b = b \\oplus a",
                   check_add_commutative)
    run_test_cases(test_cases, "a \\ominus b = -(b \\ominus a)",
                   check_subtract_sort_of_commutative)
    run_test_cases(test_cases, "0 \\ominus a = -a",
                   check_0_minus_a_negates)
    run_test_cases(test_cases, "(a \\oplus b) \\ominus a = b",
                   check_a_plus_b_minus_a_left_assoc)
    run_test_cases(test_cases, "a \\oplus (b \\ominus a) = b",
                   check_a_plus_b_minus_a_right_assoc)
    run_test_cases(test_cases, "alternate method for calculate magnitude of u ominus v",
                   check_alternate_method_to_calculate_u_ominus_v)
    run_test_cases(test_cases, "|u ominus v| = |v ominus u|",
                   check_u_ominus_v_equals_v_ominus_u)
    run_test_cases(test_cases, "|u oplus v| = |v oplus u|",
                   check_u_oplus_v_equals_v_oplus_u)


if __name__ == "__main__":
    main()
