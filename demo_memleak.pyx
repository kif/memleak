import cython
import sys
from cpython.ref cimport PyObject, Py_XDECREF
from cython cimport view
import numpy
cimport numpy
from libc.string cimport memset,memcpy

cdef struct lut_point:
    numpy.int32_t idx
    numpy.float32_t coef
dtype_lut=numpy.dtype([("idx",numpy.int32),("coef",numpy.float32)])


class DemoLeak(object):
    """
    Demo of the memory 
    """

    @cython.boundscheck(False)
    def __init__(self,dim0=1000, dim1=1000):
        self.lut = None
        self.dim0=dim0
        self.dim1=dim1
        self.need_decref = sys.version_info<(2,7)

    def init_lut(self):
        cdef lut_point[:,:] lut       
        lut = view.array(shape=(self.dim0, self.dim1),
                        itemsize=sizeof(lut_point), format="if")
        cdef numpy.ndarray[numpy.float64_t, ndim=2] tmp_ary = numpy.empty(shape=(self.dim0, self.dim1), dtype=numpy.float64)
        memcpy(&tmp_ary[0,0], &lut[0,0], self.dim0*self.dim1*sizeof(lut_point))
        self.lut = tmp_ary.view(dtype=dtype_lut)
#        lut = numpy.recarray(shape=(self.dim0, self.dim1),dtype=[("idx",numpy.int32),("coef",numpy.float32)])
#        memset(<void*>&lut[0,0],0,lut.nbytes)
#        self.lut = lut       
        
        print("Refcount: %s"%sys.getrefcount(self.lut))
        
        
    def use_lut(self):
        cdef int rc_before, rc_after
        cdef bint need_decref = False
        rc_before = sys.getrefcount(self.lut)
        cdef lut_point[:,:] lut = self.lut
        rc_after = sys.getrefcount(self.lut)
        need_decref = self.need_decref &  ((rc_after-rc_before)>=2)
        


        if need_decref:
            print("Decref needed", self.need_decref, rc_after, rc_before, need_decref)
            Py_XDECREF(<PyObject *> self.lut)

