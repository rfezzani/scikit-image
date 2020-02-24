import pytest
import numpy as np
from scipy import ndimage as ndi
from ...draw.draw3d import ellipsoid
from ..._shared import testing
from ..._shared.testing import (assert_equal, assert_almost_equal,
                                assert_allclose)
from .._moments import (moments, moments_central, moments_coords,
                        moments_coords_central, moments_normalized,
                        moments_hu, centroid, inertia_tensor,
                        inertia_tensor_eigvals)


def test_moments():
    image = np.zeros((20, 20), dtype=np.double)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    m = moments(image)
    assert_equal(m[0, 0], 3)
    assert_almost_equal(m[1, 0] / m[0, 0], 14.5)
    assert_almost_equal(m[0, 1] / m[0, 0], 14.5)


def test_moments_central():
    image = np.zeros((20, 20), dtype=np.double)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    mu = moments_central(image, (14.5, 14.5))

    # check for proper centroid computation
    mu_calc_centroid = moments_central(image)
    assert_equal(mu, mu_calc_centroid)

    # shift image by dx=2, dy=2
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[16, 16] = 1
    image2[17, 17] = 1
    image2[16, 17] = 0.5
    image2[17, 16] = 0.5
    mu2 = moments_central(image2, (14.5 + 2, 14.5 + 2))
    # central moments must be translation invariant
    assert_equal(mu, mu2)


def test_moments_coords():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu_image = moments(image)

    coords = np.array([[r, c] for r in range(13, 17)
                       for c in range(13, 17)], dtype=np.double)
    mu_coords = moments_coords(coords)
    assert_almost_equal(mu_coords, mu_image)


def test_moments_central_coords():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu_image = moments_central(image, (14.5, 14.5))

    coords = np.array([[r, c] for r in range(13, 17)
                       for c in range(13, 17)], dtype=np.double)
    mu_coords = moments_coords_central(coords, (14.5, 14.5))
    assert_almost_equal(mu_coords, mu_image)

    # ensure that center is being calculated normally
    mu_coords_calc_centroid = moments_coords_central(coords)
    assert_almost_equal(mu_coords_calc_centroid, mu_coords)

    # shift image by dx=3 dy=3
    image = np.zeros((20, 20), dtype=np.double)
    image[16:20, 16:20] = 1
    mu_image = moments_central(image, (14.5, 14.5))

    coords = np.array([[r, c] for r in range(16, 20)
                       for c in range(16, 20)], dtype=np.double)
    mu_coords = moments_coords_central(coords, (14.5, 14.5))
    assert_almost_equal(mu_coords, mu_image)


def test_moments_normalized():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu = moments_central(image, (14.5, 14.5))
    nu = moments_normalized(mu)
    # shift image by dx=-3, dy=-3 and scale by 0.5
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[11:13, 11:13] = 1
    mu2 = moments_central(image2, (11.5, 11.5))
    nu2 = moments_normalized(mu2)
    # central moments must be translation and scale invariant
    assert_almost_equal(nu, nu2, decimal=1)


def test_moments_normalized_3d():
    image = ellipsoid(1, 1, 10)
    mu_image = moments_central(image)
    nu = moments_normalized(mu_image)
    assert nu[0, 0, 2] > nu[0, 2, 0]
    assert_almost_equal(nu[0, 2, 0], nu[2, 0, 0])

    coords = np.where(image)
    mu_coords = moments_coords_central(coords)
    assert_almost_equal(mu_coords, mu_image)


def test_moments_normalized_invalid():
    with testing.raises(ValueError):
        moments_normalized(np.zeros((3, 3)), 3)
    with testing.raises(ValueError):
        moments_normalized(np.zeros((3, 3)), 4)


def test_moments_hu():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:15, 13:17] = 1
    mu = moments_central(image, (13.5, 14.5))
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    # shift image by dx=2, dy=3, scale by 0.5 and rotate by 90deg
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[11, 11:13] = 1
    image2 = image2.T
    mu2 = moments_central(image2, (11.5, 11))
    nu2 = moments_normalized(mu2)
    hu2 = moments_hu(nu2)
    # central moments must be translation and scale invariant
    assert_almost_equal(hu, hu2, decimal=1)


@pytest.mark.parametrize("dtype_in", [np.int8, np.int16, np.int32, np.int64,
                                      np.uint8, np.uint16, np.uint32,
                                      np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("dtype_out", [np.float32, np.float64])
def test_moments_hu_dtype(dtype_in, dtype_out):
    img = np.zeros((20, 20), dtype=dtype_in)

    # default dtype parameter
    if dtype_in == np.float32:
        assert moments_hu(img).dtype == np.float32
    else:
        assert moments_hu(img).dtype == np.float64

    # explicit dtype setting
    assert moments_hu(img, dtype=dtype_out).dtype == dtype_out


def test_centroid():
    image = np.zeros((20, 20), dtype=np.double)
    image[14, 14:16] = 1
    image[15, 14:16] = 1/3
    image_centroid = centroid(image)
    assert_allclose(image_centroid, (14.25, 14.5))


def test_inertia_tensor_2d():
    image = np.zeros((40, 40))
    image[15:25, 5:35] = 1  # big horizontal rectangle (aligned with axis 1)
    T = inertia_tensor(image)
    assert T[0, 0] > T[1, 1]
    np.testing.assert_allclose(T[0, 1], 0)
    v0, v1 = inertia_tensor_eigvals(image, T=T)
    np.testing.assert_allclose(np.sqrt(v0/v1), 3, rtol=0.01, atol=0.05)


def test_inertia_tensor_3d():
    image = ellipsoid(10, 5, 3)
    T0 = inertia_tensor(image)
    eig0, V0 = np.linalg.eig(T0)
    # principal axis of ellipse = eigenvector of smallest eigenvalue
    v0 = V0[:, np.argmin(eig0)]

    assert np.allclose(v0, [1, 0, 0]) or np.allclose(-v0, [1, 0, 0])

    imrot = ndi.rotate(image.astype(float), 30, axes=(0, 1), order=1)
    Tr = inertia_tensor(imrot)
    eigr, Vr = np.linalg.eig(Tr)
    vr = Vr[:, np.argmin(eigr)]

    # Check that axis has rotated by expected amount
    pi, cos, sin = np.pi, np.cos, np.sin
    R = np.array([[ cos(pi/6), -sin(pi/6), 0],
                  [ sin(pi/6),  cos(pi/6), 0],
                  [         0,          0, 1]])
    expected_vr = R @ v0
    assert (np.allclose(vr, expected_vr, atol=1e-3, rtol=0.01) or
            np.allclose(-vr, expected_vr, atol=1e-3, rtol=0.01))
