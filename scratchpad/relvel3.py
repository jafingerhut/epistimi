#! /usr/bin/env python3

import numpy as np
import random

c=299792458.0


def lorentz_gamma(v):
    v2 = np.dot(v, v)
    return 1.0 / np.sqrt(1.0 - v2 / c**2)


# Calculate and return v_B \oplus u
# This is done according to one of 3 equations, all 3 proven
# algebraically to give the same result, in the Appendix section
# titled "Relativistic velocity addition in 3 dimensions" in my
# scratchpad document.
def add(v_B, u):
    normvB2 = np.dot(v_B, v_B)
    v_B_dot_u = np.dot(v_B, u)
    gamma_B = lorentz_gamma(v_B)
    factor = 1.0 / (gamma_B * (1.0 + v_B_dot_u / (c**2)))
    return factor * (u + (((gamma_B-1) * v_B_dot_u / normvB2) + gamma_B) * v_B)


# Calculate and return v_A \ominus v_B
# This is done by calculating the parallel and perpendicular
# comoponents of the relativistic velocity difference, then adding
# those two vectors together.  You can find equations for this in the
# Appendix section titled "Relativistic velocity addition in 3
# dimensions" in my scratchpad document.
def subtract(v_A, v_B):
    v_A_dot_V_B = np.dot(v_A, v_B)
    normvB2 = np.dot(v_B, v_B)
    gamma_B = lorentz_gamma(v_B)
    v_A_parallel = (v_A_dot_V_B / normvB2) * v_B
    v_A_perp = v_A - v_A_parallel
    denom = 1 - (v_A_dot_V_B / (c**2))
    u_prime_parallel = (v_A_parallel - v_B) / denom
    u_prime_perp = (v_A_perp / gamma_B) / denom
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


def main():
    test_cases = [
        {'v_A': np.array([0.0, -0.3 * c, 0.0]),
         'v_B': np.array([0.0,  0.5 * c, 0.0])
         },
        {'v_A': np.array([0.1 * c, 0.0, 0.0]),
         'v_B': np.array([0.0, 0.1 * c, 0.0])
         },
        {'v_A': np.array([0.5 * c, 0.0, 0.0]),
         'v_B': np.array([0.0, 0.5 * c, 0.0])
         },
        {'v_A': np.array([0.4 * c, 0.2 * c, 0.0]),   # B at 0.4c
         'v_B': np.array([-0.1 * c, -0.3 * c, 0.0])  # C at -0.1c
         },
        {'v_A': np.array([0.9 * c, 0.1 * c, 0.1 * c]),
         'v_B': np.array([-0.4 * c, 0.7 * c, 0.2 * c])
         },
    ]
    for test_case in test_cases:
        v_A = test_case['v_A']
        v_B = test_case['v_B']
        sum1 = add(v_A, v_B)
        sum2 = add(v_B, v_A)
        err = abs_diff(sum1, sum2)
        print("--------------------")
        if err > 0.0001:
            print("v_A=%s v_B=%s add is NOT commutative, err=%s" % (v_A, v_B, err))
            print("v_A oplus v_B=%s" % (sum1))
            print("v_B oplus v_A=%s" % (sum2))
        else:
            print("v_A=%s v_B=%s add is commutative, err=%s v_A oplus v_B=%s" % (v_A, v_B, err, sum1))


if __name__ == "__main__":
    main()
