#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from .._shared.fused_numerics cimport np_floats

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)
    bint npy_isnan(float x)


cdef inline np_floats _get_fraction(np_floats from_value, np_floats to_value,
                                    np_floats level):
    if (to_value == from_value):
        return 0
    return ((level - from_value) / (to_value - from_value))


cdef inline tuple _top(Py_ssize_t r0, Py_ssize_t c0, np_floats ul,
                       np_floats ur, np_floats level):
    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    return dtype(r0), dtype(c0 + _get_fraction(ul, ur, level))


cdef inline tuple _bottom(Py_ssize_t r1, Py_ssize_t c0, np_floats ll,
                          np_floats lr, np_floats level):
    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    return dtype(r1), dtype(c0 + _get_fraction(ll, lr, level))


cdef inline tuple _left(Py_ssize_t r0, Py_ssize_t c0, np_floats ul,
                        np_floats ll, np_floats level):
    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    return dtype(r0 + _get_fraction(ul, ll, level)), dtype(c0)


cdef inline tuple _right(Py_ssize_t r0, Py_ssize_t c1, np_floats ur,
                         np_floats lr, np_floats level):
    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    return dtype(r0 + _get_fraction(ur, lr, level)), dtype(c1)


def iterate_and_store(np_floats[:, :] array, np_floats level,
                      Py_ssize_t vertex_connect_high,
                      cnp.uint8_t[:, :] mask):
    """Iterate across the given array in a marching-squares fashion,
    looking for segments that cross 'level'. If such a segment is
    found, its coordinates are added to a growing list of segments,
    which is returned by the function.  if vertex_connect_high is
    nonzero, high-values pixels are considered to be face+vertex
    connected into objects; otherwise low-valued pixels are.

    Positions where the boolean array ``mask`` is ``False`` are considered
    as not containing data.
    """

    cdef Py_ssize_t n_row = array.shape[0], n_col = array.shape[1]

    if n_row < 2 or n_col < 2:
        raise ValueError("Input array must be at least 2x2.")

    cdef list arc_list = []

    # The plan is to iterate a 2x2 square across the input array. This means
    # that the upper-left corner of the square needs to iterate across a
    # sub-array that's one-less-large in each direction (so that the square
    # never steps out of bounds). The square is represented by four pointers:
    # ul, ur, ll, and lr (for 'upper left', etc.). We also maintain the current
    # 2D coordinates for the position of the upper-left pointer.

    cdef unsigned char square_case
    cdef np_floats ul, ur, ll, lr
    cdef Py_ssize_t r0, r1, c0, c1

    for r0 in range(n_row-1):
        r1 = r0 + 1
        for c0 in range(n_col-1):
            # There are sixteen different possible square types,
            # diagramed below.  A + indicates that the vertex is above
            # the contour value, and a - indicates that the vertex is
            # below or equal to the contour value.  The vertices of
            # each square are:
            # ul ur
            # ll lr
            # and can be treated as a binary value with the bits in
            # that order. Thus each square case can be numbered:
            #  0--   1+-   2-+   3++   4--   5+-   6-+   7++
            #   --    --    --    --    +-    +-    +-    +-
            #
            #  8--   9+-  10-+  11++  12--  13+-  14-+  15++
            #   -+    -+    -+    -+    ++    ++    ++    ++
            #
            # The position of the line segment that cuts through (or
            # doesn't, in case 0 and 15) each square is clear, except
            # in cases 6 and 9. In this case, where the segments are
            # placed is determined by vertex_connect_high.  If
            # vertex_connect_high is false, then lines like \\ are
            # drawn through square 6, and lines like // are drawn
            # through square 9.  Otherwise, the situation is reversed.
            # Finally, recall that we draw the lines so that (moving
            # from tail to head) the lower-valued pixels are on the
            # left of the line. So, for example, case 1 entails a line
            # slanting from the middle of the top of the square to the
            # middle of the left side of the square.

            c1 = c0 + 1

            # Skip this square if any of the four input values are masked out.
            if not(mask is None or (mask[r0, c0] and mask[r0, c1] and
                                    mask[r1, c0] and mask[r1, c1])):
                continue

            ul = array[r0, c0]
            ur = array[r0, c1]
            ll = array[r1, c0]
            lr = array[r1, c1]

            # Skip this square if any of the four input values are NaN.
            if npy_isnan(ul) or npy_isnan(ur) or npy_isnan(ll) or npy_isnan(lr):
                continue

            square_case = 0
            if (ul > level): square_case += 1
            if (ur > level): square_case += 2
            if (ll > level): square_case += 4
            if (lr > level): square_case += 8

            if (square_case == 0 or square_case == 15):
                # only do anything if there's a line passing through the
                # square. Cases 0 and 15 are entirely below/above the contour.
                continue

            if (square_case == 1):
                # top to left
                arc_list.append(_top(r0, c0, ul, ur, level))
                arc_list.append(_left(r0, c0, ul, ll, level))
            elif (square_case == 2):
                # right to top
                arc_list.append(_right(r0, c1, ur, lr, level))
                arc_list.append(_top(r0, c0, ul, ur, level))
            elif (square_case == 3):
                # right to left
                arc_list.append(_right(r0, c1, ur, lr, level))
                arc_list.append(_left(r0, c0, ul, ll, level))
            elif (square_case == 4):
                # left to bottom
                arc_list.append(_left(r0, c0, ul, ll, level))
                arc_list.append(_bottom(r1, c0, ll, lr, level))
            elif (square_case == 5):
                # top to bottom
                arc_list.append(_top(r0, c0, ul, ur, level))
                arc_list.append(_bottom(r1, c0, ll, lr, level))
            elif (square_case == 6):
                if vertex_connect_high:
                    arc_list.append(_left(r0, c0, ul, ll, level))
                    arc_list.append(_top(r0, c0, ul, ur, level))

                    arc_list.append(_right(r0, c1, ur, lr, level))
                    arc_list.append(_bottom(r1, c0, ll, lr, level))
                else:
                    arc_list.append(_right(r0, c1, ur, lr, level))
                    arc_list.append(_top(r0, c0, ul, ur, level))
                    arc_list.append(_left(r0, c0, ul, ll, level))
                    arc_list.append(_bottom(r1, c0, ll, lr, level))
            elif (square_case == 7):
                # right to bottom
                arc_list.append(_right(r0, c1, ur, lr, level))
                arc_list.append(_bottom(r1, c0, ll, lr, level))
            elif (square_case == 8):
                # bottom to right
                arc_list.append(_bottom(r1, c0, ll, lr, level))
                arc_list.append(_right(r0, c1, ur, lr, level))
            elif (square_case == 9):
                if vertex_connect_high:
                    arc_list.append(_top(r0, c0, ul, ur, level))
                    arc_list.append(_right(r0, c1, ur, lr, level))

                    arc_list.append(_bottom(r1, c0, ll, lr, level))
                    arc_list.append(_left(r0, c0, ul, ll, level))
                else:
                    arc_list.append(_top(r0, c0, ul, ur, level))
                    arc_list.append(_left(r0, c0, ul, ll, level))

                    arc_list.append(_bottom(r1, c0, ll, lr, level))
                    arc_list.append(_right(r0, c1, ur, lr, level))
            elif (square_case == 10):
                # bottom to top
                arc_list.append(_bottom(r1, c0, ll, lr, level))
                arc_list.append(_top(r0, c0, ul, ur, level))
            elif (square_case == 11):
                # bottom to left
                arc_list.append(_bottom(r1, c0, ll, lr, level))
                arc_list.append(_left(r0, c0, ul, ll, level))
            elif (square_case == 12):
                # lef to right
                arc_list.append(_left(r0, c0, ul, ll, level))
                arc_list.append(_right(r0, c1, ur, lr, level))
            elif (square_case == 13):
                # top to right
                arc_list.append(_top(r0, c0, ul, ur, level))
                arc_list.append(_right(r0, c1, ur, lr, level))
            elif (square_case == 14):
                # left to top
                arc_list.append(_left(r0, c0, ul, ll, level))
                arc_list.append(_top(r0, c0, ul, ur, level))

    return arc_list
