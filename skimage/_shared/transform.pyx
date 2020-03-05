#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp


cdef cnp.int64_tint integrate(np_anyint[:, ::1] sat,
                              Py_ssize_t r0, Py_ssize_t c0,
                              Py_ssize_t r1, Py_ssize_t c1) nogil:
    """
    Using a summed area table / integral image, calculate the sum
    over a given window.

    This function is the same as the `integrate` function in
    `skimage.transform.integrate`, but this Cython version significantly
    speeds up the code.

    Parameters
    ----------
    sat : ndarray of np_real_numeric
        Summed area table / integral image.
    r0, c0 : Py_ssize_t
        Top-left corner of block to be summed.
    r1, c1 : Py_ssize_t
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : np_signed_numeric
        Sum over the given window.
    """

    cdef int S = 0

    S += sat[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= sat[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[r1, c0 - 1]

    return S


cdef np_floats integrate(np_floats[:, ::1] sat,
                         Py_ssize_t r0, Py_ssize_t c0,
                         Py_ssize_t r1, Py_ssize_t c1) nogil:
    """
    Using a summed area table / integral image, calculate the sum
    over a given window.

    This function is the same as the `integrate` function in
    `skimage.transform.integrate`, but this Cython version significantly
    speeds up the code.

    Parameters
    ----------
    sat : ndarray of np_real_numeric
        Summed area table / integral image.
    r0, c0 : Py_ssize_t
        Top-left corner of block to be summed.
    r1, c1 : Py_ssize_t
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : np_signed_numeric
        Sum over the given window.
    """

    cdef np_floats S = 0

    S += sat[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= sat[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[r1, c0 - 1]

    return S
