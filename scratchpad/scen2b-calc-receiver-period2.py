#! /usr/bin/env python3

import numpy as np
import random

c=299792458.0


def lorentz_gamma(v):
    v2 = np.dot(v, v)
    return 1.0 / np.sqrt(1.0 - v2 / c**2)


def compute_delta_tau_C_n(v_B, v_C, n, c=299792458.0):
    """
    Compute the proper time interval on C's clock between receiving
    the n-th and (n+1)-th light pulses emitted from B at 1-second intervals
    in B's frame.

    Parameters:
        v_B: np.array, shape (3,), velocity vector of B in lab frame
        v_C: np.array, shape (3,), velocity vector of C in lab frame
        n: int, pulse index (n >= 1)
        c: float, speed of light (default in m/s)

    Returns:
        delta_tau_C: float, proper time interval between receptions on C's clock
    """

    def compute_reception_time(n_emit, v_B, v_C, c):
        gamma_B = lorentz_gamma(v_B)
        tau = n_emit * gamma_B

        a = v_C
        b = v_B
        A = np.dot(a, a) - c**2
        B = -2 * tau * (np.dot(a, b) - c**2)
        C = tau**2 * (np.dot(b, b) - c**2)

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            raise ValueError(f"Pulse {n_emit}: no real solution (discriminant < 0)")

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B + sqrt_disc) / (2 * A)
        t2 = (-B - sqrt_disc) / (2 * A)

        t_emit = tau
        t_receive = max(t for t in (t1, t2) if t > t_emit)
        return t_receive

    if n < 1:
        raise ValueError("Pulse index n must be at least 1")

    gamma_C = lorentz_gamma(v_C)
    t_n = compute_reception_time(n, v_B, v_C, c)
    t_np1 = compute_reception_time(n + 1, v_B, v_C, c)

    delta_tau_C = (t_np1 - t_n) / gamma_C
    return delta_tau_C


# Compute time interval measured by receiver between pulses,
# using method in Appendix D.1 with intermediate value G.
def reception_time2(v_B, v_C, T_B):
    v_B_dot_v_C = np.dot(v_B, v_C)
    v_C_mag_sq = np.dot(v_C, v_C)
    gamma_B = lorentz_gamma(v_B)
    gamma_C = lorentz_gamma(v_C)
    v_C_minus_v_B_mag_sq = np.dot(v_C - v_B, v_C - v_B)
    v_C_mag_sq_minus_v_B_dot_v_C = v_C_mag_sq - v_B_dot_v_C
    G = (v_C_mag_sq_minus_v_B_dot_v_C**2) + (c**2 - v_C_mag_sq) * v_C_minus_v_B_mag_sq
    delta = (gamma_B * T_B / gamma_C) * (1 + ((gamma_C / c)**2) * (v_C_mag_sq_minus_v_B_dot_v_C + np.sqrt(G)))
    return delta


# Calculate and return v_A \ominus v_B
def relativistic_velocity_subtract(v_A, v_B):
    v_A_dot_V_B = np.dot(v_A, v_B)
    v_B_mag_sq = np.dot(v_B, v_B)
    gamma_B = lorentz_gamma(v_B)
    v_A_parallel = (v_A_dot_V_B / v_B_mag_sq) * v_B
    v_A_perp = v_A - v_A_parallel
    denom = 1 - (v_A_dot_V_B / (c**2))
    u_prime_parallel = (v_A_parallel - v_B) / denom
    u_prime_perp = (v_A_perp / gamma_B) / denom
    return u_prime_parallel + u_prime_perp

def random_rescale_if_faster_than_c(v):
    mag = np.sqrt(np.dot(v, v))
    if mag < c:
        return v
    print("v >= c, rescaling it so its direction is same, but its magnitude is a uniform random value in the range [0,c)")
    new_beta = random.random()
    v = (v / mag) * c * new_beta
    mag = np.sqrt(np.dot(v, v))
    if mag >= c:
        print(f"BUG in program.  New magnitude after rescaling is > c.  v={v} mag={mag}")
        sys.exit(1)
    return v

# Velocities in m/s (e.g., B and C move in opposite directions along x-axis)
#v_B = np.array([0.4 * 299792458, 0.0, 0.0])   # B at 0.4c
#v_C = np.array([-0.1 * 299792458, 0.0, 0.0])  # C at -0.1c

test_cases = [
    {'v_B': np.array([0.4 * c, 0.2 * c, 0.0]),   # B at 0.4c
     'v_C': np.array([-0.1 * c, -0.3 * c, 0.0])  # C at -0.1c
     },
    {'v_B': np.array([0.9 * c, 0.1 * c, 0.1 * c]),
     'v_C': np.array([-0.4 * c, 0.7 * c, 0.2 * c])
     },
    ]

for n in range(10):
    test_case = {'v_B': np.array([random.random() * c, random.random() * c, random.random() * c]),
                 'v_C': np.array([random.random() * c, random.random() * c, random.random() * c])}
    test_cases.append(test_case)

count = 0
for test_case in test_cases:
    count += 1
    v_B = test_case['v_B']
    v_C = test_case['v_C']

    print("----------------------------------------")
    print(f"Test case #{count}")
    print("----------------------------------------")
    v_B = random_rescale_if_faster_than_c(v_B)
    v_C = random_rescale_if_faster_than_c(v_C)
    v_C_minus_v_B = v_B - v_C
    print(f"v_B x={v_B[0]} y={v_B[1]} z={v_B[2]}")
    print(f"v_C x={v_C[0]} y={v_C[1]} z={v_C[2]}")
    print(f"v_C_minus_v_B x={v_C_minus_v_B[0]} y={v_C_minus_v_B[1]} z={v_C_minus_v_B[2]}")

    beta_B = np.sqrt(np.dot(v_B, v_B)) / c
    gamma_B = lorentz_gamma(v_B)
    print(f"beta_B={beta_B}")
    print(f"gamma_B={gamma_B}")
    if beta_B >= 1.0:
        print("beta_B >= 1.0, so skipping this test case")
        continue

    beta_C = np.sqrt(np.dot(v_C, v_C)) / c
    gamma_C = lorentz_gamma(v_C)
    print(f"beta_C={beta_C}")
    print(f"gamma_C={gamma_C}")
    if beta_C >= 1.0:
        print("beta_C >= 1.0, rescaling it so its direction is same, but its magnitude is a uniform random value in the range [0,c)")
        continue

    # Compute interval between reception of 5th and 6th pulses
    for n in range(1,6):
        delta_tau_C = compute_delta_tau_C_n(v_B, v_C, n)
        print(f"Time on C's clock between receiving pulses {n} and {n+1}: {delta_tau_C:.9f} s")

    T_B = 1.0
    delta2 = reception_time2(v_B, v_C, T_B)
    print(f"delta2={delta2:.9f} s")

    diff = relativistic_velocity_subtract(v_C, v_B)
    diff_mag = np.sqrt(np.dot(diff, diff))
    beta_diff = diff_mag / c
    print(f"beta_diff={beta_diff}")
    D = np.sqrt((1+beta_diff) / (1-beta_diff))
    delta3 = D * T_B
    print(f"delta3={delta3:.9f} s")

    ratio = delta2 / delta3
    print(f"ratio=delta2 / delta3={delta2/delta3}")
    if abs(ratio - 1.0) > 0.0001:
        print(f"PROBLEM?  delta2 / delta3 significantly different than 1")
