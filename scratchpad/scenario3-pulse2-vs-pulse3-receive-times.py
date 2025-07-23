#! /usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 Andy Fingerhut (andy.fingerhut@gmail.com)
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""
Generate a plot for Scenario 3 of Andy Fingerhut's scratchpad
document, showing for a given value of beta the curves of locations
where an observer at rest relative to rod R will receive a pulse from
event 3 k seconds after it receives the pulse from event 2 (if k is
positive -- pulse reception order is the opposite of that for negative
k).
""")

parser.add_argument('--beta', dest='beta', type=float,
                    help="""The speed of rod S relative to rod R,
                    as a fraction of the speed of light.""")
args, remaining_args = parser.parse_known_args()

if not args.beta:
    print("Must provide option --beta", file=sys.stderr)
    parser.print_help()
    sys.exit(1)
if args.beta <= 0.0 or args.beta >= 1.0:
    print("Value for option beta was %.3f.  It must be in range (0, 1)"
          "" % (args.beta))
    parser.print_help()
    sys.exit(1)

# Dimensions of plot
Xmin = -5
Xmax = 5
Ymin = -5
Ymax = 5

# Length of rod L, in light-seconds
L = 1

# Define constants for positions of left and right ends of rod R
l = np.array([-L/2, 0])
r = np.array([L/2, 0])

# Define beta values and compute corresponding constant values
beta = args.beta
gamma = 1/np.sqrt(1-beta**2)
event_2_time = L/(gamma*beta)
event_3_time = L/beta
ev_3_minus_ev_2_time = (L * beta) / (1 + np.sqrt(1 - beta**2))
k_max = L + ev_3_minus_ev_2_time
k_min = -L + ev_3_minus_ev_2_time
print("beta=%s" % (beta))
print("gamma=%s" % (gamma))
print("event_2_time=%s" % (event_2_time))
print("event_3_time=%s" % (event_3_time))
print("ev_3_minus_ev_2_time = (L * beta) / (1 + np.sqrt(1 - beta**2)) = %s"
      "" % (ev_3_minus_ev_2_time))
print("k_min=%s" % (k_min))
print("k_max=%s" % (k_max))
k_values = [ k_min+0.01*(k_max-k_min),
             k_min+0.1*(k_max-k_min) ]
maybe_B_delta_receive_time = -ev_3_minus_ev_2_time
if maybe_B_delta_receive_time < k_min:
    maybe_B_delta_receive_time = k_min + 0.001 * (k_max - k_min)
k_values += [maybe_B_delta_receive_time]
k_values += [ 0,
              ev_3_minus_ev_2_time,
              k_max-0.1*(k_max-k_min),
              k_max-0.01*(k_max-k_min),
             ]
diff_values = [k - ev_3_minus_ev_2_time for k in k_values]

# Generate X,Y grid
X = np.linspace(Xmin, Xmax, 600)
Y = np.linspace(Ymin, Ymax, 600)
XX, YY = np.meshgrid(X, Y)

# Compute distances to points l and r
d_l = np.sqrt((XX - l[0])**2 + (YY - l[1])**2)
d_r = np.sqrt((XX - r[0])**2 + (YY - r[1])**2)

# Create figure
plt.figure(figsize=(7, 7))

# Plot each curve
colors = []
while len(colors) < len(k_values):
    colors += ['blue', 'green', 'orange', 'red', 'purple']

#print("len(k_values)=%d" % (len(k_values)))
#print("len(colors)=%d" % (len(colors)))
#print("colors=%s" % (colors))

for k, diff, color in zip(k_values, diff_values, colors):
    print("k=%s diff=%s color=%s" % (k, diff, color))
    # Compute mask for the level set d_l - d_r = k
    CS = plt.contour(XX, YY, d_l - d_r, levels=[diff], colors=color)
    label = "k=%.2f" % (k)
    fmt = {CS.levels[0]: label}
    if diff < 0:
        manual_label_location = [(-2*L, 2*L)]
    else:
        manual_label_location = [(2*L, 2*L)]
    plt.clabel(CS, fmt=fmt, fontsize=10, inline=True,
               manual=manual_label_location)

# Mark points l and r
plt.plot(l[0], l[1], 'ko', label='Point l (%.2f,0) (event 3)' % (-L/2))
plt.plot(r[0], r[1], 'ks', label='Point r (%.2f,0) (event 2)' % (L/2))

# Labels and styling
plt.xlabel("X")
plt.ylabel("Y")
plt.title(r"$\beta=%.2f$ event 3 %.2f sec after 2; Curves where pulse 3 received $k$ sec after 2"
          "" % (beta, ev_3_minus_ev_2_time))
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(Xmin, Xmax)
plt.ylim(Ymin, Ymax)
plt.legend(loc='upper left')
plt.gca().set_aspect('equal')

plt.tight_layout()
fname = ""
plt.savefig("pulse-receive-deltas-L-%.1f-beta-%.2f.pdf"
            "" % (L, beta),
            format='pdf')
plt.show()
