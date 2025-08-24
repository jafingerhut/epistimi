#! /usr/bin/env python3

import numpy as np
import random

c=299792458.0


def lorentz_gamma(v):
    v2 = np.dot(v, v)
    return 1.0 / np.sqrt(1.0 - v2 / c**2)


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
    gamma_B = lorentz_gamma(vB)
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
    gamma_B = lorentz_gamma(vB)
    vA_parallel = (vA_dot_vB / normvB2) * vB
    vA_perp = vA - vA_parallel
    denom = 1 - (vA_dot_vB / (c**2))
    u_prime_parallel = (vA_parallel - vB) / denom
    u_prime_perp = (vA_perp / gamma_B) / denom
    return u_prime_parallel + u_prime_perp



def random_rescale_if_faster_than_c(v):
    mag1 = np.sqrt(np.dot(v, v))
    if mag1 < c:
        return v
    new_beta = random.random()
    v = (v / mag1) * c * new_beta
    mag2 = np.sqrt(np.dot(v, v))
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
    run_test_cases(test_cases, "a \oplus b = b \oplus a",
                   check_add_commutative)
    run_test_cases(test_cases, "a \ominus b = -(b \ominus a)",
                   check_subtract_sort_of_commutative)
    run_test_cases(test_cases, "0 \ominus a = -a",
                   check_0_minus_a_negates)
    run_test_cases(test_cases, "(a \oplus b) \ominus a = b",
                   check_a_plus_b_minus_a_left_assoc)
    run_test_cases(test_cases, "a \oplus (b \ominus a) = b",
                   check_a_plus_b_minus_a_right_assoc)


if __name__ == "__main__":
    main()
