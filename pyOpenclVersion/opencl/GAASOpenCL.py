
#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import os
import matplotlib.pyplot as plt

class Gaas_OCL_API:
    def __init__(self, oclDevice = None) -> None:
        #init opencl program
        # os.environ["PYOPENCL_CTX"] = '0:1'
        # os.environ["PYOPENCL_NO_CACHE"] = '1'
        
        if(oclDevice == None):
            platform = cl.get_platforms()
            self.dev = platform[0].get_devices(device_type=cl.device_type.GPU)
            print("GAAS OpenCL: Using ",self.dev)
        else:
            self.dev = oclDevice

        self.ctx = cl.Context(devices=self.dev)
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        thisFilePath = os.path.realpath(os.path.dirname(__file__))

        try:
            file = open(os.path.join(thisFilePath,"Gaas.c"))
            srcStr = file.read()
            file.close()
        except IOError:
            print("Failed to load opencl source files")
            exit(-1)

        self.prg = cl.Program(self.ctx, srcStr)
        # os.environ['GMX_GPU_DISABLE_COMPATIBILITY_CHECK'] = '1'
        self.prg.build(options="-I"+thisFilePath + " -w")
        print(self.prg.get_build_info(self.ctx.devices[0],cl.program_build_info.LOG))
        print(self.prg.get_info(cl.program_info.KERNEL_NAMES))

    def runVoigtTest(self, wvn : np.array):
        wvn = wvn.astype(np.float64)
        wvn_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=wvn)
        out_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, wvn.nbytes)
        knl = self.prg.voigtTest
        knl(self.queue,wvn.shape,None,out_g,wvn_g)
        out_np = np.empty_like(wvn)
        cl.enqueue_copy(self.queue, out_np, out_g)
        return out_np