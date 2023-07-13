# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:35:19 2023

@author: gholt
"""

# Code for Exercise 1
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots() # Create a figure containing a single axes.

# Exercise 1
ax.plot([1, 2, 3, 4], [1, 4, 2, 3]) # Plot some data on the axes.

# Exercise 1
fig = plt.figure() # an empty figure with no Axes
fig, ax = plt.subplots() # a figure with a single Axes
fig, axs = plt.subplots(2, 2) # a figure with a 2x2 grid of Axes

# Exercise 2 Tutorial Code
x = np.linspace(0, 2, 100)
fig, ax = plt.subplots() # Create a figure and an axes.
ax.plot(x, x, label='linear') # Plot some data on the axes.
ax.plot(x, x**2, label='quadratic') # Plot more data on the axes...
ax.plot(x, x**3, label='cubic') # ... and some more.
ax.set_xlabel('x label') # Add an x-label to the axes.
ax.set_ylabel('y label') # Add a y-label to the axes.
ax.set_title("Simple Plot") # Add a title to the axes.
ax.legend() # Add a legend.


# Exercise 2
x = np.linspace(0, 50)
fig, ax = plt.subplots() # Create a figure and an axes.
fig.suptitle("Exercise 2")
ax.plot(x, x*2, label='Cost') # Plot some data on the axes.
ax.plot(x, x+20, label='Revenue') # Plot more data on the axes...
ax.set_xlabel('Cost') # Add an x-label to the axes.
ax.set_ylabel('Dollars ($)') # Add a y-label to the axes.
ax.set_title("Cost-Revenue") # Add a title to the axes.
ax.legend() # Add a legend.
