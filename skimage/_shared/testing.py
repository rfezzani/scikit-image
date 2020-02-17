"""
Testing utilities.
"""

import os
import re
import struct
import threading
import functools
from tempfile import NamedTemporaryFile

import pytest
import numpy as np
from numpy import testing
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_array_less, assert_array_almost_equal_nulp,
                           assert_equal, TestCase, assert_allclose,
                           assert_almost_equal, assert_, assert_warns,
                           assert_no_warnings)

import warnings

from ..io._io import imread, imsave
from ..data import chelsea, moon
from ..util.dtype import img_as_uint, img_as_float, img_as_int, img_as_ubyte
from ._warnings import expected_warnings


SKIP_RE = re.compile(r"(\s*>>>.*?)(\s*)#\s*skip\s+if\s+(.*)$")

skipif = pytest.mark.skipif
xfail = pytest.mark.xfail
parametrize = pytest.mark.parametrize
raises = pytest.raises
fixture = pytest.fixture

# true if python is running in 32bit mode
# Calculate the size of a void * pointer in bits
# https://docs.python.org/3/library/struct.html
arch32 = struct.calcsize("P") * 8 == 32


def assert_less(a, b, msg=None):
    message = "%r is not lower than %r" % (a, b)
    if msg is not None:
        message += ": " + msg
    assert a < b, message


def assert_greater(a, b, msg=None):
    message = "%r is not greater than %r" % (a, b)
    if msg is not None:
        message += ": " + msg
    assert a > b, message


def doctest_skip_parser(func):
    """ Decorator replaces custom skip test markup in doctests

    Say a function has a docstring::

        >>> something, HAVE_AMODULE, HAVE_BMODULE = 0, False, False
        >>> something # skip if not HAVE_AMODULE
        0
        >>> something # skip if HAVE_BMODULE
        0

    This decorator will evaluate the expression after ``skip if``.  If this
    evaluates to True, then the comment is replaced by ``# doctest: +SKIP``. If
    False, then the comment is just removed. The expression is evaluated in the
    ``globals`` scope of `func`.

    For example, if the module global ``HAVE_AMODULE`` is False, and module
    global ``HAVE_BMODULE`` is False, the returned function will have docstring::

        >>> something # doctest: +SKIP
        >>> something + else # doctest: +SKIP
        >>> something # doctest: +SKIP

    """
    lines = func.__doc__.split('\n')
    new_lines = []
    for line in lines:
        match = SKIP_RE.match(line)
        if match is None:
            new_lines.append(line)
            continue
        code, space, expr = match.groups()

        try:
            # Works as a function decorator
            if eval(expr, func.__globals__):
                code = code + space + "# doctest: +SKIP"
        except AttributeError:
            # Works as a class decorator
            if eval(expr, func.__init__.__globals__):
                code = code + space + "# doctest: +SKIP"

        new_lines.append(code)
    func.__doc__ = "\n".join(new_lines)
    return func


def roundtrip(image, plugin, suffix):
    """Save and read an image using a specified plugin"""
    if '.' not in suffix:
        suffix = '.' + suffix
    temp_file = NamedTemporaryFile(suffix=suffix, delete=False)
    fname = temp_file.name
    temp_file.close()
    imsave(fname, image, plugin=plugin)
    new = imread(fname, plugin=plugin)
    try:
        os.remove(fname)
    except Exception:
        pass
    return new


def color_check(plugin, fmt='png'):
    """Check roundtrip behavior for color images.

    All major input types should be handled as ubytes and read
    back correctly.
    """
    img = img_as_ubyte(chelsea())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)

    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2.astype(np.uint8), r2)

    img3 = img_as_float(img)
    r3 = roundtrip(img3, plugin, fmt)
    testing.assert_allclose(r3, img)

    img4 = img_as_int(img)
    if fmt.lower() in (('tif', 'tiff')):
        img4 -= 100
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img4)
    else:
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img_as_ubyte(img4))

    img5 = img_as_uint(img)
    r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img)


def mono_check(plugin, fmt='png'):
    """Check the roundtrip behavior for images that support most types.

    All major input types should be handled.
    """

    img = img_as_ubyte(moon())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)

    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2.astype(np.uint8), r2)

    img3 = img_as_float(img)
    r3 = roundtrip(img3, plugin, fmt)
    if r3.dtype.kind == 'f':
        testing.assert_allclose(img3, r3)
    else:
        testing.assert_allclose(r3, img_as_uint(img))

    img4 = img_as_int(img)
    if fmt.lower() in (('tif', 'tiff')):
        img4 -= 100
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img4)
    else:
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img_as_uint(img4))

    img5 = img_as_uint(img)
    r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img5)


def setup_test():
    """Default package level setup routine for skimage tests.

    Import packages known to raise warnings, and then
    force warnings to raise errors.

    Also set the random seed to zero.
    """
    warnings.simplefilter('default')

    from scipy import signal, ndimage, special, optimize, linalg
    from scipy.io import loadmat
    from skimage import viewer

    np.random.seed(0)

    warnings.simplefilter('error')


def teardown_test():
    """Default package level teardown routine for skimage tests.

    Restore warnings to default behavior
    """
    warnings.simplefilter('default')


def test_parallel(num_threads=2, warnings_matching=None):
    """Decorator to run the same function multiple times in parallel.

    This decorator is useful to ensure that separate threads execute
    concurrently and correctly while releasing the GIL.

    Parameters
    ----------
    num_threads : int, optional
        The number of times the function is run in parallel.

    warnings_matching: list or None
        This parameter is passed on to `expected_warnings` so as not to have
        race conditions with the warnings filters. A single
        `expected_warnings` context manager is used for all threads.
        If None, then no warnings are checked.

    """

    assert num_threads > 0

    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            with expected_warnings(warnings_matching):
                threads = []
                for i in range(num_threads - 1):
                    thread = threading.Thread(target=func, args=args,
                                              kwargs=kwargs)
                    threads.append(thread)
                for thread in threads:
                    thread.start()

                result = func(*args, **kwargs)

                for thread in threads:
                    thread.join()

                return result

        return inner

    return wrapper


if __name__ == '__main__':
    color_check('pil')
    mono_check('pil')
    mono_check('pil', 'bmp')
    mono_check('pil', 'tiff')
