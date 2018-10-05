#! python2.7
from __future__ import division
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pprint
pp = pprint.PrettyPrinter(indent=4)


def entropy_i(prob_i):
    return -prob_i * math.log(prob_i, 2)


prob = [i / 100 for i in range(0, 100)]
entropy = [entropy_i(p) if p else 0 for p in prob]

plt.plot(prob, entropy)
plt.xlabel("$probability$")
plt.ylabel("-$p_i * log_{2}p_{i}$")
plt.show()
