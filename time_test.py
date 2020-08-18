import time
import numpy as np

batch_size = 32
timestep_max = 16
runtimes = 1000

begin = time.time()
for _ in range(runtimes):
    rands = np.random.randint(timestep_max, size = batch_size*2)
    for i in range(batch_size):
        bx = np.random.random((batch_size, timestep_max))
        timestep = (rands[i*2]%(timestep_max-1))+1
        bx[i, : timestep] = np.random.random((timestep,))
print((time.time() - begin) * 1000)

begin = time.time()
for _ in range(runtimes):
    bx = np.random.random((batch_size, timestep_max))
    rands = np.random.randint(timestep_max, size = batch_size*2)
    for i in range(batch_size):
        timestep = (rands[i*2]%(timestep_max-1))+1
        bx[i, : timestep] = np.random.random((timestep,))
        bx[i, timestep:] = 0
print((time.time() - begin) * 1000)