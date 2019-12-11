"""
Random walker segmentation algorithm

from *Random walks for image segmentation*, Leo Grady, IEEE Trans
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.
"""

import numpy as np
from scipy import sparse, ndimage as ndi

from .._shared.utils import warn

# executive summary for next code block: try to import umfpack from
# scipy, but make sure not to raise a fuss if it fails since it's only
# needed to speed up a few cases.
# See discussions at:
# https://groups.google.com/d/msg/scikit-image/FrM5IGP6wh4/1hp-FtVZmfcJ
# https://stackoverflow.com/questions/13977970/ignore-exceptions-printed-to-stderr-in-del/13977992?noredirect=1#comment28386412_13977992
try:
    from scipy.sparse.linalg.dsolve import umfpack
    old_del = umfpack.UmfpackContext.__del__

    def new_del(self):
        try:
            old_del(self)
        except AttributeError:
            pass
    umfpack.UmfpackContext.__del__ = new_del
    UmfpackContext = umfpack.UmfpackContext()
except Exception:
    UmfpackContext = None

try:
    from pyamg import ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False

from ..util import img_as_float
from ..filters import rank_order

from scipy.sparse.linalg import cg
import scipy
from distutils.version import LooseVersion as Version
import functools

if Version(scipy.__version__) >= Version('1.1'):
    cg = functools.partial(cg, atol=0)

# -----------Laplacian--------------------


def _make_graph_edges_3d(n_x, n_y, n_z):
    """Returns a list of edges for a 3D image.

    Parameters
    ----------
    n_x: integer
        The size of the grid in the x direction.
    n_y: integer
        The size of the grid in the y direction
    n_z: integer
        The size of the grid in the z direction

    Returns
    -------
    edges : (2, N) ndarray
        with the total number of edges::

            N = n_x * n_y * (nz - 1) +
                n_x * (n_y - 1) * nz +
                (n_x - 1) * n_y * nz

        Graph edges with each column describing a node-id pair.
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[..., :-1].ravel(),
                            vertices[..., 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges


def _compute_weights_3d(data, spacing, beta=130, eps=1.e-6,
                        multichannel=False):
    # Weight calculation is main difference in multispectral version
    # Original gradient**2 replaced with sum of gradients ** 2
    gradients = 0
    for channel in range(data.shape[-1]):
        gradients += np.concatenate(
            [np.abs(np.diff(data[..., channel], axis=ax)).ravel() / spacing[ax]
             for ax in [2, 1, 0]], axis=0) ** 2
    # All channels considered together in this standard deviation
    scale_factor = -1 / (10 * data.std())
    if multichannel:
        # New final term in beta to give == results in trivial case where
        # multiple identical spectra are passed.
        scale_factor /= np.sqrt(data.shape[-1])
    weights = np.exp(scale_factor * beta * gradients)
    weights += eps
    return weights


def _make_laplacian_sparse(edges, weights):
    """
    Sparse implementation
    """
    pixel_nb = edges.shape[1]
    i_indices = edges.ravel()
    j_indices = edges[::-1].ravel()
    data = - np.hstack((weights, weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)),
                            shape=(pixel_nb, pixel_nb))
    lap.setdiag(-np.ravel(lap.sum(axis=1)))
    return lap.tocsr()


def _clean_labels_ar(X, labels, copy=False):
    if copy:
        labels = np.copy(labels)
    labels = np.ravel(labels)
    labels[labels == 0] = X.astype(labels.dtype)
    return labels


def _build_linear_system(data, spacing, labels, mask,
                         beta, multichannel):
    """
    Build the matrix A and rhs B of the linear system to solve.
    A and B are two block of the laplacian of the image graph.
    """
    lap_sparse = _build_laplacian(data, spacing, mask=mask,
                                  beta=beta, multichannel=multichannel)

    if mask is None:
        labels = labels.ravel().astype('int32')
    else:
        labels = labels[mask].astype('int32')

    indices = np.arange(labels.size)
    cond = labels > 0
    unlabeled_indices = indices[~cond]
    seeds_indices = indices[cond]
    # The following two lines take most of the time in this function
    rows = lap_sparse[unlabeled_indices, :]
    B = rows[:, seeds_indices]
    lap_sparse = rows[:, unlabeled_indices]
    nlabels = labels.max()
    seeds = labels[seeds_indices]

    mask = np.vstack([seeds == lab for lab in range(1, nlabels+1)])
    rhs = B * sparse.csc_matrix(mask).transpose()

    return lap_sparse, rhs


def _solve_linear_system(lap_sparse, B, tol, return_full_prob, mode):
    if mode == 'cg_mg' and not amg_loaded:
        warn("pyamg (http://pyamg.github.io/)) is needed to use "
             "this mode, but is not installed. The 'cg' mode will be "
             "used instead.")
        mode = 'cg'

    if mode == 'cg':
        lap_sparse = lap_sparse.tocsc()
        X = [cg(lap_sparse, -B[:, i].toarray(), tol=tol)[0]
             for i in range(B.shape[1])]
    elif mode == 'cg_mg':
        ml = ruge_stuben_solver(lap_sparse)
        M = ml.aspreconditioner(cycle='V')
        X = [cg(lap_sparse, -B[:, i].toarray(), tol=tol, M=M, maxiter=30)[0]
             for i in range(B.shape[1])]
    elif mode == 'bf':
        lap_sparse = lap_sparse.tocsc()
        solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = solver(-B.toarray()).T

    if not return_full_prob:
        X = np.array(X)
        X = X.argmax(axis=0)
    return X


def _mask_edges_weights(edges, weights, mask):
    """
    Remove edges of the graph connected to masked nodes, as well as
    corresponding weights of the edges.
    """
    mask0 = np.hstack((mask[..., :-1].ravel(), mask[:, :-1].ravel(),
                       mask[:-1].ravel()))
    mask1 = np.hstack((mask[..., 1:].ravel(), mask[:, 1:].ravel(),
                       mask[1:].ravel()))
    ind_mask = np.logical_and(mask0, mask1)
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    max_node_index = edges.max()
    # Reassign edges labels to 0, 1, ... edges_number - 1
    order = np.searchsorted(np.unique(edges.ravel()),
                            np.arange(max_node_index + 1))
    edges = order[edges.astype(np.int64)]
    return edges, weights


def _build_laplacian(data, spacing, mask=None, beta=50,
                     multichannel=False):
    l_x, l_y, l_z = data.shape[:3]
    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, spacing, beta=beta, eps=1.e-10,
                                  multichannel=multichannel)
    if mask is not None:
        edges, weights = _mask_edges_weights(edges, weights, mask)
    lap = _make_laplacian_sparse(edges, weights)
    del edges, weights
    return lap


def _check_isolated_seeds(labels):
    """
    Prune isolated seed pixels to prevent labeling errors, and
    return coordinates and label values of isolated seeds, so
    that it is possible to put labels back in random walker output.
    """
    fill = ndi.binary_propagation(labels == 0, mask=(labels >= 0))
    isolated = np.logical_and(labels > 0, np.logical_not(fill))
    inds = np.nonzero(isolated)
    values = labels[inds]
    labels[inds] = -1
    return inds, values


def _unchanged_labels(labels, return_full_prob=False):
    """
    Return the input array of labels, unless ``return_full_prob`` is True,
    in which case a mask of pixels is returned for each unique label, with the
    last dimension corresponding to unique labels.
    """
    if return_full_prob:
        return np.concatenate([np.atleast_3d(labels == lab)
                               for lab in np.unique(labels) if lab > 0],
                              axis=-1)
    return labels


# ----------- Random walker algorithm --------------------------------


def random_walker(data, labels, beta=130, mode='bf', tol=1.e-3, copy=True,
                  multichannel=False, return_full_prob=False, spacing=None):
    """Random walker algorithm for segmentation from markers.

    Random walker algorithm is implemented for gray-level or multichannel
    images.

    Parameters
    ----------
    data : array_like
        Image to be segmented in phases. Gray-level `data` can be two- or
        three-dimensional; multichannel data can be three- or four-
        dimensional (multichannel=True) with the highest dimension denoting
        channels. Data spacing is assumed isotropic unless the `spacing`
        keyword argument is used.
    labels : array of ints, of same shape as `data` without channels dimension
        Array of seed markers labeled with different positive integers
        for different phases. Zero-labeled pixels are unlabeled pixels.
        Negative labels correspond to inactive pixels that are not taken
        into account (they are removed from the graph). If labels are not
        consecutive integers, the labels array will be transformed so that
        labels are consecutive. In the multichannel case, `labels` should have
        the same shape as a single channel of `data`, i.e. without the final
        dimension denoting channels.
    beta : float, optional
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).
    mode : string, available options {'cg_mg', 'cg', 'bf'}
        Mode for solving the linear system in the random walker algorithm.
        If no preference given, automatically attempt to use the fastest
        option available ('cg_mg' from pyamg >> 'cg' with UMFPACK > 'bf').

        - 'bf' (brute force): an LU factorization of the Laplacian is
          computed. This is fast for small images (<1024x1024), but very slow
          and memory-intensive for large images (e.g., 3-D volumes).
        - 'cg' (conjugate gradient): the linear system is solved iteratively
          using the Conjugate Gradient method from scipy.sparse.linalg. This is
          less memory-consuming than the brute force method for large images,
          but it is quite slow.
        - 'cg_mg' (conjugate gradient with multigrid preconditioner): a
          preconditioner is computed using a multigrid solver, then the
          solution is computed with the Conjugate Gradient method.  This mode
          requires that the pyamg module (http://pyamg.github.io/) is
          installed. For images of size > 512x512, this is the recommended
          (fastest) mode.

    tol : float, optional
        tolerance to achieve when solving the linear system, in
        cg' and 'cg_mg' modes.
    copy : bool, optional
        If copy is False, the `labels` array will be overwritten with
        the result of the segmentation. Use copy=False if you want to
        save on memory.
    multichannel : bool, optional
        If True, input data is parsed as multichannel data (see 'data' above
        for proper input format in this case).
    return_full_prob : bool, optional
        If True, the probability that a pixel belongs to each of the labels
        will be returned, instead of only the most likely label.
    spacing : iterable of floats, optional
        Spacing between voxels in each spatial dimension. If `None`, then
        the spacing between pixels/voxels in each dimension is assumed 1.

    Returns
    -------
    output : ndarray
        * If `return_full_prob` is False, array of ints of same shape as
          `data`, in which each pixel has been labeled according to the marker
          that reached the pixel first by anisotropic diffusion.
        * If `return_full_prob` is True, array of floats of shape
          `(nlabels, data.shape)`. `output[label_nb, i, j]` is the probability
          that label `label_nb` reaches the pixel `(i, j)` first.

    See also
    --------
    skimage.morphology.watershed: watershed segmentation
        A segmentation algorithm based on mathematical morphology
        and "flooding" of regions from markers.

    Notes
    -----
    Multichannel inputs are scaled with all channel data combined. Ensure all
    channels are separately normalized prior to running this algorithm.

    The `spacing` argument is specifically for anisotropic datasets, where
    data points are spaced differently in one or more spatial dimensions.
    Anisotropic data is commonly encountered in medical imaging.

    The algorithm was first proposed in [1]_.

    The algorithm solves the diffusion equation at infinite times for
    sources placed on markers of each phase in turn. A pixel is labeled with
    the phase that has the greatest probability to diffuse first to the pixel.

    The diffusion equation is solved by minimizing x.T L x for each phase,
    where L is the Laplacian of the weighted graph of the image, and x is
    the probability that a marker of the given phase arrives first at a pixel
    by diffusion (x=1 on markers of the phase, x=0 on the other markers, and
    the other coefficients are looked for). Each pixel is attributed the label
    for which it has a maximal value of x. The Laplacian L of the image
    is defined as:

       - L_ii = d_i, the number of neighbors of pixel i (the degree of i)
       - L_ij = -w_ij if i and j are adjacent pixels

    The weight w_ij is a decreasing function of the norm of the local gradient.
    This ensures that diffusion is easier between pixels of similar values.

    When the Laplacian is decomposed into blocks of marked and unmarked
    pixels::

        L = M B.T
            B A

    with first indices corresponding to marked pixels, and then to unmarked
    pixels, minimizing x.T L x for one phase amount to solving::

        A x = - B x_m

    where x_m = 1 on markers of the given phase, and 0 on other markers.
    This linear system is solved in the algorithm using a direct method for
    small images, and an iterative method for larger images.

    References
    ----------
    .. [1] Leo Grady, Random walks for image segmentation, IEEE Trans Pattern
        Anal Mach Intell. 2006 Nov;28(11):1768-83.
        :DOI:`10.1109/TPAMI.2006.233`.

    Examples
    --------
    >>> np.random.seed(0)
    >>> a = np.zeros((10, 10)) + 0.2 * np.random.rand(10, 10)
    >>> a[5:8, 5:8] += 1
    >>> b = np.zeros_like(a)
    >>> b[3, 3] = 1  # Marker for first phase
    >>> b[6, 6] = 2  # Marker for second phase
    >>> random_walker(a, b)
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)

    """
    # Parse input data
    if mode is None:
        if amg_loaded:
            mode = 'cg_mg'
        elif UmfpackContext is not None:
            mode = 'cg'
        else:
            mode = 'bf'
    elif mode not in ('cg_mg', 'cg', 'bf'):
        raise ValueError("{mode} is not a valid mode. Valid modes are 'cg_mg',"
                         " 'cg' and 'bf'".format(mode=mode))

    if UmfpackContext is None and mode == 'cg':
        warn('"cg" mode will be used, but it may be slower than '
             '"bf" because SciPy was built without UMFPACK. Consider'
             ' rebuilding SciPy with UMFPACK; this will greatly '
             'accelerate the conjugate gradient ("cg") solver. '
             'You may also install pyamg and run the random_walker '
             'function in "cg_mg" mode (see docstring).')

    label_values, renum = np.unique(labels, return_inverse=True)

    if not (label_values == 0).any():
        warn('Random walker only segments unlabeled areas, where '
             'labels == 0. No zero valued areas in labels were '
             'found. Returning provided labels.')

        return _unchanged_labels(labels, return_full_prob)

    # This algorithm expects 4-D arrays of floats, where the first three
    # dimensions are spatial and the final denotes channels. 2-D images have
    # a singleton placeholder dimension added for the third spatial dimension,
    # and single channel images likewise have a singleton added for channels.
    # The following block ensures valid input and coerces it to the correct
    # form.
    if not multichannel:
        if data.ndim not in (2, 3):
            raise ValueError('For non-multichannel input, data must be of '
                             'dimension 2 or 3.')
        dims = data.shape  # To reshape final labeled result
        data = np.atleast_3d(img_as_float(data))[..., np.newaxis]
    else:
        if data.ndim not in (3, 4):
            raise ValueError('For multichannel input, data must have 3 or 4 '
                             'dimensions.')
        dims = data[..., 0].shape  # To reshape final labeled result
        data = img_as_float(data)
        if data.ndim == 3:  # 2D multispectral, needs singleton in 3rd axis
            data = data[:, :, np.newaxis, :]

    # Spacing kwarg checks
    if spacing is None:
        spacing = np.ones(3)
    elif len(spacing) == len(dims):
        if len(spacing) == 2:  # Need a dummy spacing for singleton 3rd dim
            spacing = np.r_[spacing, 1.]
        else:                  # Convert to array
            spacing = np.asarray(spacing)
    else:
        raise ValueError('Input argument `spacing` incorrect, should be an '
                         'iterable with one number per spatial dimension.')

    if copy:
        labels = np.copy(labels)

    # If some labeled pixels are isolated inside pruned zones, prune them
    # as well and keep the labels for the final output
    # inds_isolated_seeds, isolated_values = _check_isolated_seeds(labels)

    null_mask = labels == 0
    pos_mask = labels > 0
    mask = labels >= 0

    fill = ndi.binary_propagation(null_mask, mask=mask)
    isolated = np.logical_and(pos_mask, np.logical_not(fill))

    inds_isolated_seeds = np.nonzero(isolated)
    isolated_values = labels[inds_isolated_seeds]

    labels[inds_isolated_seeds] = isolated_values
    pos_mask[inds_isolated_seeds] = False

    # If the array has pruned zones, be sure that no isolated pixels
    # exist between pruned zones (they could not be determined)
    if label_values[0] < 0 or np.any(isolated):
        isolated = np.logical_and(
            np.logical_not(ndi.binary_propagation(pos_mask, mask=mask)),
            null_mask)

        labels[isolated] = -1
        if np.all(isolated[null_mask]):
            warn('Random walker only segments unlabeled areas, where '
                 'labels == 0. No zero valued areas in labels were '
                 'found. Returning provided labels.')
            return _unchanged_labels(labels, return_full_prob)

        mask[isolated] = False
        mask = np.atleast_3d(mask)
    else:
        mask = None

    # Reorder label values to have consecutive integers (no gaps)
    labels = np.atleast_3d(renum.reshape(dims) + label_values[0])

    lap_sparse, B = _build_linear_system(data, spacing, labels, mask,
                                         beta, multichannel)

    # We solve the linear system
    # lap_sparse X = B
    # where X[i, j] is the probability that a marker of label i arrives
    # first at pixel j by anisotropic diffusion.
    X = _solve_linear_system(lap_sparse, B, tol, return_full_prob, mode)

    # Clean up results
    if return_full_prob:
        labels = labels.astype(np.float)
        # Put back labels of isolated seeds
        if len(isolated_values) > 0:
            labels[inds_isolated_seeds] = isolated_values
        X = np.array([_clean_labels_ar(Xline, labels, copy=True).reshape(dims)
                      for Xline in X])
        for i in range(1, int(label_values[-1]) + 1):
            mask_i = np.squeeze(labels == i)
            X[:, mask_i] = 0
            X[i - 1, mask_i] = 1
    else:
        X = _clean_labels_ar(X + 1, labels).reshape(dims)
        # Put back labels of isolated seeds
        X[inds_isolated_seeds] = isolated_values
    return X
