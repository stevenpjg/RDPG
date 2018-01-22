# -*- coding: utf-8 -*-
"""
This script takes episodes stored in episode_reward.txt and displays the learning curve.
"""
import numpy as np
import matplotlib.pyplot as plt
lines = np.loadtxt("episode_reward.txt", comments="#", delimiter="\n", unpack=False)
plt.plot(lines)
plt.show()
