#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: language=c++

import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector

from .._shared.fused_numerics cimport np_real_numeric
from .._shared.transform cimport integrate


FEATURE_TYPE = {'type-2-x': 0, 'type-2-y': 1,
                'type-3-x': 2, 'type-3-y': 3,
                'type-4': 4}

N_RECTANGLE = {'type-2-x': 2, 'type-2-y': 2,
               'type-3-x': 3, 'type-3-y': 3,
               'type-4': 4}


cdef vector[vector[Rectangle]] _type_2_x_feature(Py_ssize_t width,
                                                 Py_ssize_t height) nogil:
    """Compute the coordinates of type-2-x Haar-like features.

    Parameters
    ----------
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    """
    cdef:
        Rectangle single_rect
        vector[vector[Rectangle]] rect_feat
        Py_ssize_t x, y, dx, dy

    rect_feat = vector[vector[Rectangle]](2)

    for y in range(height):
        for x in range(width):
            for dy in range(height - max(y, 1)):
                for dx in range((width - x) // 2):
                    set_rectangle_feature(&single_rect,
                                          y, x,
                                          y + dy, x + dx)
                    rect_feat[0].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y, x + dx + 1,
                                          y + dy,
                                          x + 2 * dx + 1)
                    rect_feat[1].push_back(single_rect)
    return rect_feat


cdef vector[vector[Rectangle]] _type_2_y_feature(Py_ssize_t width,
                                                 Py_ssize_t height) nogil:
    """Compute the coordinates of type-2-y Haar-like features.

    Parameters
    ----------
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    """
    cdef:
        Rectangle single_rect
        vector[vector[Rectangle]] rect_feat
        Py_ssize_t x, y, dx, dy

    rect_feat = vector[vector[Rectangle]](2)

    for y in range(height):
        for x in range(width):
            for dy in range((height - y) // 2):
                for dx in range(width - max(x, 1)):
                    set_rectangle_feature(&single_rect,
                                          y, x,
                                          y + dy, x + dx)
                    rect_feat[0].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y + dy + 1, x,
                                          y + 2 * dy + 1, x + dx)
                    rect_feat[1].push_back(single_rect)
    return rect_feat


cdef vector[vector[Rectangle]] _type_3_x_feature(Py_ssize_t width,
                                                 Py_ssize_t height) nogil:
    """Compute the coordinates of type-3-x Haar-like features.

    Parameters
    ----------
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    """
    cdef:
        Rectangle single_rect
        vector[vector[Rectangle]] rect_feat
        Py_ssize_t x, y, dx, dy

    rect_feat = vector[vector[Rectangle]](3)
    for y in range(height):
        for x in range(width):
            for dy in range(height - max(y, 1)):
                for dx in range((width - x) // 3):
                    set_rectangle_feature(&single_rect,
                                          y, x,
                                          y + dy, x + dx)
                    rect_feat[0].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y, x + dx + 1,
                                          y + dy, x + 2 * dx + 1)
                    rect_feat[1].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y, x + 2 * dx + 2,
                                          y + dy, x + 3 * dx + 2)
                    rect_feat[2].push_back(single_rect)
    return rect_feat


cdef vector[vector[Rectangle]] _type_3_y_feature(Py_ssize_t width,
                                                 Py_ssize_t height) nogil:
    """Compute the coordinates of type-3-y Haar-like features.

    Parameters
    ----------
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    """
    cdef:
        Rectangle single_rect
        vector[vector[Rectangle]] rect_feat
        Py_ssize_t x, y, dx, dy

    rect_feat = vector[vector[Rectangle]](3)
    for y in range(height):
        for x in range(width):
            for dy in range((height - y) // 3):
                for dx in range(width - max(1, x)):
                    set_rectangle_feature(&single_rect,
                                          y, x,
                                          y + dy, x + dx)
                    rect_feat[0].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y + dy + 1, x,
                                          y + 2 * dy + 1, x + dx)
                    rect_feat[1].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y + 2 * dy + 2, x,
                                          y + 3 * dy + 2, x + dx)
                    rect_feat[2].push_back(single_rect)
    return rect_feat


cdef vector[vector[Rectangle]] _type_4_feature(Py_ssize_t width,
                                               Py_ssize_t height) nogil:
    """Compute the coordinates of type-4 Haar-like features.

    Parameters
    ----------
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    """
    cdef:
        Rectangle single_rect
        vector[vector[Rectangle]] rect_feat
        Py_ssize_t x, y, dx, dy

    rect_feat = vector[vector[Rectangle]](4)
    for y in range(height):
        for x in range(width):
            for dy in range((height - y) // 2):
                for dx in range((width - x) // 2):
                    set_rectangle_feature(&single_rect,
                                          y, x,
                                          y + dy, x + dx)
                    rect_feat[0].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y, x + dx + 1,
                                          y + dy, x + 2 * dx + 1)
                    rect_feat[1].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y + dy + 1, x,
                                          y + 2 * dy + 1, x + dx)
                    rect_feat[3].push_back(single_rect)
                    set_rectangle_feature(&single_rect,
                                          y + dy + 1, x + dx + 1,
                                          y + 2 * dy + 1,
                                          x + 2 * dx + 1)
                    rect_feat[2].push_back(single_rect)
    return rect_feat


cdef vector[vector[Rectangle]] _haar_like_feature_coord(
    Py_ssize_t width,
    Py_ssize_t height,
    unsigned int feature_type) nogil:
    """Private function to compute the coordinates of all Haar-like features.
    """
    cdef:
        vector[vector[Rectangle]] rect_feat

    if feature_type == 0:
        # type -> 2 rectangles split along x axis
        rect_feat = _type_2_x_feature(width, height)
    elif feature_type == 1:
        # type -> 2 rectangles split along y axis
        rect_feat = _type_2_y_feature(width, height)
    elif feature_type == 2:
        # type -> 3 rectangles split along x axis
        rect_feat = _type_3_x_feature(width, height)
    elif feature_type == 3:
        # type -> 3 rectangles split along y axis
        rect_feat = _type_3_y_feature(width, height)
    elif feature_type == 4:
        # type -> 4 rectangles split along x and y axis
        rect_feat = _type_4_feature(width, height)

    return rect_feat


cpdef inline _rect2list(Rectangle rect):
    return [(rect.top_left.row, rect.top_left.col),
            (rect.bottom_right.row, rect.bottom_right.col)]


cpdef haar_like_feature_coord_wrapper(width, height, feature_type):
    """Compute the coordinates of Haar-like features.

    Parameters
    ----------
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    feature_type : str
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

    Returns
    -------
    feature_coord : (n_features, n_rectangles, 2, 2), ndarray of list of \
tuple coord
        Coordinates of the rectangles for each feature.
    feature_type : (n_features,), ndarray of str
        The corresponding type for each feature.

    """
    cdef:
        vector[vector[Rectangle]] rect
        Py_ssize_t n_rectangle, n_feature
        Py_ssize_t i, j
        # cast the height and width to the right type
        Py_ssize_t height_win = <Py_ssize_t> height
        Py_ssize_t width_win = <Py_ssize_t> width

    rect = _haar_like_feature_coord(width_win, height_win,
                                    FEATURE_TYPE[feature_type])
    n_feature = rect[0].size()
    n_rectangle = rect.size()

    # allocate the output based on the number of rectangle
    output = np.empty((n_feature,), dtype=object)
    for j in range(n_feature):
        coord_feature = []
        for i in range(n_rectangle):
            coord_feature.append(_rect2list(rect[i][j]))
        output[j] = coord_feature

    return output, np.array([feature_type] * n_feature, dtype=object)


cdef np_real_numeric[:, ::1] _haar_like_feature(
        np_real_numeric[:, ::1] int_image,
        vector[vector[Rectangle]] coord,
        Py_ssize_t n_rectangle, Py_ssize_t n_feature):
    """Private function releasing the GIL to compute the integral for the
    different rectangle."""
    cdef:
        np_real_numeric[:, ::1] rect_feature = np.empty(
            (n_rectangle, n_feature), dtype=int_image.base.dtype)

        Py_ssize_t idx_rect, idx_feature

    with nogil:
        for idx_rect in range(n_rectangle):
            for idx_feature in range(n_feature):
                rect_feature[idx_rect, idx_feature] = integrate(
                    int_image,
                    coord[idx_rect][idx_feature].top_left.row,
                    coord[idx_rect][idx_feature].top_left.col,
                    coord[idx_rect][idx_feature].bottom_right.row,
                    coord[idx_rect][idx_feature].bottom_right.col)

    return rect_feature


cpdef haar_like_feature_wrapper(
    cnp.ndarray[np_real_numeric, ndim=2] int_image,
    r, c, width, height, feature_type, feature_coord):
    """Compute the Haar-like features for a region of interest (ROI) of an
    integral image.

    Haar-like features have been successfully used for image classification and
    object detection [1]_. It has been used for real-time face detection
    algorithm proposed in [2]_.

    Parameters
    ----------
    int_image : (M, N) ndarray
        Integral image for which the features need to be computed.
    r : int
        Row-coordinate of top left corner of the detection window.
    c : int
        Column-coordinate of top left corner of the detection window.
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    feature_type : str
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

    Returns
    -------
    haar_features : (n_features,) ndarray
        Resulting Haar-like features. Each value is equal to the subtraction of
        sums of the positive and negative rectangles. The data type depends of
        the data type of `int_image`: `int` when the data type of `int_image`
        is `uint` or `int` and `float` when the data type of `int_image` is
        `float`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Haar-like_feature
    .. [2] Oren, M., Papageorgiou, C., Sinha, P., Osuna, E., & Poggio, T.
           (1997, June). Pedestrian detection using wavelet templates.
           In Computer Vision and Pattern Recognition, 1997. Proceedings.,
           1997 IEEE Computer Society Conference on (pp. 193-199). IEEE.
           http://tinyurl.com/y6ulxfta
           :DOI:`10.1109/CVPR.1997.609319`
    .. [3] Viola, Paul, and Michael J. Jones. "Robust real-time face
           detection." International journal of computer vision 57.2
           (2004): 137-154.
           http://www.merl.com/publications/docs/TR2004-043.pdf
           :DOI:`10.1109/CVPR.2001.990517`

    """
    cdef:
        vector[vector[Rectangle]] coord
        Py_ssize_t n_rectangle, n_feature
        Py_ssize_t idx_rect, idx_feature
        np_real_numeric[:, ::1] rect_feature
        # FIXME: currently cython does not support read-only memory views.
        # Those are used with joblib when using Parallel. Therefore, we use
        # ndarray as input. We take a copy of this ndarray to create a memory
        # view to be able to release the GIL in some later processing.
        # Check the following issue to check the status of read-only memory
        # views in cython:
        # https://github.com/cython/cython/issues/1605 to be resolved
        np_real_numeric[:, ::1] int_image_memview = int_image[
            r : r + height, c : c + width].copy()

    if feature_coord is None:
        # compute all possible coordinates with a specific type of feature
        coord = _haar_like_feature_coord(width, height,
                                         FEATURE_TYPE[feature_type])
        n_feature = coord[0].size()
        n_rectangle = coord.size()
    else:
        # build the coordinate from the set provided
        n_rectangle = N_RECTANGLE[feature_type]
        n_feature = len(feature_coord)

        # the vector can be directly pre-allocated since that the size is known
        coord = vector[vector[Rectangle]](n_rectangle,
                                          vector[Rectangle](n_feature))

        for idx_rect in range(n_rectangle):
            for idx_feature in range(n_feature):
                set_rectangle_feature(
                    &coord[idx_rect][idx_feature],
                    feature_coord[idx_feature][idx_rect][0][0],
                    feature_coord[idx_feature][idx_rect][0][1],
                    feature_coord[idx_feature][idx_rect][1][0],
                    feature_coord[idx_feature][idx_rect][1][1])

    rect_feature = _haar_like_feature(int_image_memview,
                                      coord, n_rectangle, n_feature)

    # convert the memory view to numpy array and convert it to signed array if
    # necessary to avoid overflow during subtraction
    rect_feature_ndarray = np.asarray(rect_feature)
    data_type = rect_feature_ndarray.dtype
    if 'uint' in data_type.name:
        rect_feature_ndarray = rect_feature_ndarray.astype(
            data_type.name.replace('u', ''))

    # the rectangles with odd indices can always be subtracted to the rectangle
    # with even indices
    return (np.sum(rect_feature_ndarray[1::2], axis=0) -
            np.sum(rect_feature_ndarray[::2], axis=0))
