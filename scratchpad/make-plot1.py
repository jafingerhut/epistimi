#! /usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 Andy Fingerhut (andy.fingerhut@gmail.com)
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

D = 10

# Define the function you want to plot
def f(x):
    return (D + np.sqrt((D**2) - (4 * x**2))) / 2.0

# Generate x values
Xmin = 0
Xmax = D/2.0
npoints = 400
x = np.linspace(Xmin, Xmax, npoints)
y = f(x)

# Create the plot
plt.plot(x, y, label='y = f(x)')
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Plot of y = x^2')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

