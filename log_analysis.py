import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
file_name = sys.argv[0]
a  = np.load(file_name)
a = a['arr_0']

plt.figure(1)

qw = []
for i in range(1000):
    qw.append(i+1)

qw = np.asarray(qw)

if True:
    #plt.figure(1)
    #plt.subplot(221)

    plt.plot(a, qw, color='black')

    plt.ylabel('Average Reward')
    plt.xlabel('Episode number')
    plt.savefig('exp.png')
