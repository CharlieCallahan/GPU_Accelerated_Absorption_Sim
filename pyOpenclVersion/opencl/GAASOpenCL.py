
#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import os
import matplotlib.pyplot as plt


os.environ["PYOPENCL_CTX"] = '0:1'
os.environ["PYOPENCL_NO_CACHE"] = '1'

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

try:
    file = open("Gaas.c")
    srcStr = file.read()
    file.close()
except IOError:
    print("Failed to load opencl source files")
    exit(-1)

# print(srcStr)

prg = cl.Program(ctx, srcStr)
# print("AAA")
prg.build()
print(prg.get_build_info(ctx.devices[0],cl.program_build_info.LOG))

print(prg.all_kernels())

mf = cl.mem_flags

# print(prg.all)

# print(prg.get_build_info(None,cl.program_build_info.LOG))

print(prg.all_kernels())
def runVoigtTest(wvn : np.array):
    wvn = wvn.astype(np.float64)
    wvn_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=wvn)
    out_g = cl.Buffer(ctx, mf.WRITE_ONLY, wvn.nbytes)
    knl = prg.voigtTest
    # knl(queue,wvn.shape,None,wvn_g,out_g)

    out_np = np.empty_like(wvn)
    # cl.enqueue_copy(queue, out_np, out_g)
    return out_np


wvn = np.linspace(-5,5,100000)

ret = runVoigtTest(wvn)

plt.plot(wvn,ret)
plt.show()