import numpy as np
import matplotlib.pyplot as plt

qw = np.load('sampled_epoch_14_batch_1_time_step_0.npz')
qw2 = np.load('sampled_epoch_14_batch_1_time_step_1.npz')
te = [qw['X'], qw2['X']]

te = np.asarray(te)
rosemary = te[1,:,:] - te[0,:,:]


soa = np.hstack([te[0,:,:], 30*rosemary])
soa = soa[0:1,:]





X,Y,U,V = zip(*soa)
plt.figure()
ax = plt.gca()
ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
plt.draw()
plt.show()
