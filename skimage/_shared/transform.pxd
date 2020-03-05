from .fused_numerics cimport np_anyint, np_floats


cdef np_signed_numeric integrate(np_real_numeric[:, ::1] sat,
                                 Py_ssize_t r0, Py_ssize_t c0,
                                 Py_ssize_t r1, Py_ssize_t c1) nogil
