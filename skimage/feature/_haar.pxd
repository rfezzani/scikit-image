from libcpp.vector cimport vector
cimport numpy as cnp
from .._shared.fused_numerics cimport np_real_numeric


cdef struct Point2D:
    Py_ssize_t row
    Py_ssize_t col


cdef struct Rectangle:
    Point2D top_left
    Point2D bottom_right


cdef inline void set_rectangle_feature(Rectangle* rectangle,
                                       Py_ssize_t top_y,
                                       Py_ssize_t top_x,
                                       Py_ssize_t bottom_y,
                                       Py_ssize_t bottom_x) nogil:
    rectangle[0].top_left.row = top_y
    rectangle[0].top_left.col = top_x
    rectangle[0].bottom_right.row = bottom_y
    rectangle[0].bottom_right.col = bottom_x


cdef vector[vector[Rectangle]] _haar_like_feature_coord(
    Py_ssize_t width,
    Py_ssize_t height,
    unsigned int feature_type) nogil


cpdef haar_like_feature_coord_wrapper(width, height, feature_type)


cdef np_real_numeric[:, ::1] _haar_like_feature(
    np_real_numeric[:, ::1] int_image,
    vector[vector[Rectangle]] coord,
    Py_ssize_t n_rectangle, Py_ssize_t n_feature)


cpdef haar_like_feature_wrapper(
    cnp.ndarray[np_real_numeric, ndim=2] int_image,
    r, c, width, height, feature_type,
    feature_coord)
