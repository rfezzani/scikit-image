import numpy as np
cimport numpy as cnp
from .fused_numerics cimport np_signed_numeric, np_real_numeric, np_uints


cpdef np_signed_numeric integrate(np_signed_numeric[:, ::1] sat,
                                 Py_ssize_t r0, Py_ssize_t c0,
                                 Py_ssize_t r1, Py_ssize_t c1) nogil
