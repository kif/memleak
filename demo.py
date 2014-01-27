#!/usr/bin/python

import  sys, pyximport, Cython, numpy, os, gc
print("Python: %s" % sys.version)
print("Numpy: %s" % numpy.version.version)
print("Cython: %s" % Cython.__version__)
pyximport.install()
import demo_memleak

def get_mem():
    """
    Returns the occupied memory for memory-leak hunting in MByte
    """
    pid = os.getpid()
    if os.path.exists("/proc/%i/status" % pid):
        for l in open("/proc/%i/status" % pid):
            if l.startswith("VmRSS"):
                mem = int(l.split(":", 1)[1].split()[0]) / 1024.
    else:
        mem = 0
    return mem

d = demo_memleak.DemoLeak()
print("Const: %s" % sys.getrefcount(d))
d.init_lut()
print("init: %s" % sys.getrefcount(d))
lut_count = sys.getrefcount(d.lut)
for i in range(10):
    d.use_lut()
print("obj: %s \t lut: %s" % (sys.getrefcount(d), sys.getrefcount(d.lut)))
assert sys.getrefcount(d.lut) == lut_count
print gc.get_referents(d)
print gc.get_referents(d.lut)
################################################################################
# test destructor
################################################################################
mem_before = get_mem()
e = demo_memleak.DemoLeak()
e.init_lut()
e.use_lut()
#l = e.lut
mem_after = get_mem()
del e
mem_del = get_mem()
print mem_before, mem_after, mem_del
for i in range(10):
    d = demo_memleak.DemoLeak(1024, 1024)
    d.init_lut()
    for j in range(2):
        d.use_lut()
print("Memory leak: %sMB" % ((get_mem() - mem_del) / 10.))
#print sys.getrefcount(l), gc.get_referents(l)
