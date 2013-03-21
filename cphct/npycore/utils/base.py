#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - numpy back end functions shared by plugin and tools
# Copyright (C) 2012-2013  The Cph CT Toolbox Project lead by Brian Vinter
#
# This file is part of Cph CT Toolbox.
#
# Cph CT Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Cph CT Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
#
# -- END_HEADER ---
#

"""Cph CT Toolbox numpy back end functions shared by plugins and tools.
We separate I/O from the actual handlers so that they can be used inside apps
and in separate tools scripts."""

import os

from cphct.log import logging
from cphct.io import create_path_dir
from cphct.npycore import zeros, empty_like, iinfo, finfo, float128, \
    clip, arange, hstack, log, isinf, isnan, nonzero, cos, sin, pi

supported_proj_filters = [
    'hamming',
    'ram-lak',
    'shepp-logan',
    'cosine',
    'hann',
    'skip',
    ]


def prepare_output(shape, conf):
    """Shared helper to create output matrix for manipulation

    Parameters
    ----------
    shape : tuple of int
        Tuple with integer dimensions of output matrix.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : ndarray
        Returns output ndarray.
    """

    fdt = conf['output_data_type']
    return zeros(shape, dtype=fdt)


def normalize_array(
    data,
    normalize_min=None,
    normalize_max=None,
    out=None,
    ):
    """Shift and scale array values to span full range supported by the array
    data type. May be useful e.g. to make uniform values for image writers
    that do not automatically scale values.
    The optional normalize_min and normalize_max arguments are used to define a
    normalize range other than the default, None, which results in the largest
    possible range for the type.

    Parameters
    ----------
    data : ndarray
        Data matrix to normalize.
    normalize_min : object
        None or float value to use as minimum for normalize range
    normalize_max : object
        None or float value to use as maximum for normalize range
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        for without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.

    Returns
    -------
    output : ndarray
        Returns normalized ndarray.
    """

    float_type = float128
    dtype = data.dtype.type

    # We need to differentiate lookup for int and float types

    if dtype(1.5) == dtype(1):
        type_info = iinfo(dtype)
    else:
        type_info = finfo(dtype)
    (min_type, max_type) = (type_info.min, type_info.max)
    (min_norm, max_norm) = (min_type, max_type)

    # User supplied norm range of same data type

    if normalize_min is not None:
        min_norm = max(dtype(normalize_min), min_type)
    if normalize_max is not None:
        max_norm = min(dtype(normalize_max), max_type)

    # Unit range distribution of values

    dist_unity = data.astype(float_type)
    dist_unity -= dist_unity.min()
    dist_unity /= dist_unity.max()

    # Be careful not to introduce float overflow/underflow
    # IMPORTANT: do not simplify/optimize without testing with float128 !

    res = float_type(min_norm) + dist_unity * float_type(max_norm) \
        - dist_unity * float_type(min_norm)

    if out is None:
        out = empty_like(data)

    out[:] = res

    # Make sure casting did not truncate overflowing values.
    # We may still see minor rounding issues here for e.g. uint64

    out[res > float_type(max_norm)] = max_norm
    out[res < float_type(min_norm)] = min_norm

    return out


def clip_array(
    data,
    clip_min,
    clip_max,
    out=None,
    ):
    """Force all values in data into the range bounded by clip_min and
    clip_max. May be useful e.g. to remove outliers in images to improve
    the contrast.

    Parameters
    ----------
    data : ndarray
        Data matrix to clip.
    clip_min : object
        None or float value to use as minimum for clip range
    clip_max : object
        None or float value to use as maximum for clip range
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.
        
    Returns
    -------
    output : ndarray
        Returns clipped ndarray.
    """

    if out is None:
        out = empty_like(data)

    dtype = out.dtype.type

    if clip_min is None:
        clip_min = data.min()
    if clip_max is None:
        clip_max = data.max()

    # Use boundary values of out data type

    clip(data, dtype(clip_min), dtype(clip_max), out=out)

    return out


def dump_array(data, path_or_fd):
    """Dump all values in data as binary (bytes) in path. May be useful e.g.
    to save temporary results inside chain of preprocess or postprocess
    plugins. If path exists the data will be appended and otherwise the file
    will be created first.

    Parameters
    ----------
    data : ndarray
        Data matrix to dump.
    path_or_fd : file or str
        Open file object or path of file to append data values to

    Returns
    -------
    output : ndarray
        Returns unchanged ndarray.
    """

    if isinstance(path_or_fd, basestring):
        path = path_or_fd
        create_path_dir(path)
        if os.path.isfile(path):
            dump_fd = open(path, 'ab')
        else:
            dump_fd = open(path, 'wb')
    else:
        path = None
        dump_fd = path_or_fd
    logging.debug('dump to %s' % path_or_fd)
    data.tofile(dump_fd)
    if path is not None:
        dump_fd.close()

    return data


def flux_to_proj(
    data,
    zero_norm,
    air_norm,
    detector_shape,
    air_ref_pixel=None,
    out=None,
    ):
    """Convert stacked intensity measurement data values from intensity/flux
    to attenuation projections using logarithmic normalization with previously
    prepared air_norm and zero_norm values. This is a quite common operation
    to translate measured raw integer intensity values to actual projection
    values.

    Parameters
    ----------
    data : ndarray
        Data matrix with measured intensities.
    zero_norm : ndarray
        Matrix with background intensities.
    air_norm : ndarray
        Matrix with pure air intensities.
    detector_shape : tuple
        Detector shape as tuple with number of rows and columns.
    air_ref_pixel : str, optional
        Tuble of pixel posistion (y,x) in *data* containing air value
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.

    Returns
    -------
    output : ndarray
        Returns ndarray with normalized projections.
    """

    # Allocate out if not provided

    if out is None:
        out = empty_like(data)

    # Fake stack of projections even for a single flat projection

    orig_data_shape = data.shape
    if data.shape == detector_shape:
        data.shape = tuple([1] + list(data.shape))

    orig_out_shape = out.shape
    if out.shape == detector_shape:
        out.shape = tuple([1] + list(out.shape))

    # If air_ref_pixel is set, log_air_norm is
    # posponed until air diff is found

    if air_ref_pixel is None:
        log_air_norm = log(air_norm - zero_norm)

    for proj_idx in xrange(data.shape[0]):

        # Copy data to out and extract raw projection
        # If 'data' is 'out' numpy omits the actual copy

        out[proj_idx] = data[proj_idx]
        raw_proj = out[proj_idx]

        if air_ref_pixel:
            air_y = air_ref_pixel[0]
            air_x = air_ref_pixel[1]
            air_diff = air_norm[air_y, air_x] - raw_proj[air_y, air_x]
            log_air_norm = air_norm - air_diff - zero_norm

        raw_proj -= zero_norm
        log(raw_proj, raw_proj)
        raw_proj *= -1
        raw_proj += log_air_norm

        isinf_indexes = isinf(raw_proj)
        isnan_indexes = isnan(raw_proj)
        below_zero_indexes = raw_proj < 0.0

        raw_proj[isinf_indexes] = log_air_norm[isinf_indexes]
        raw_proj[isnan_indexes] = 0.0
        raw_proj[below_zero_indexes] = 0.0

    data.shape = orig_data_shape
    out.shape = orig_out_shape

    return out


def square_array(data, out=None):
    """Square all values in data.

    Parameters
    ----------
    data : ndarray
        Data matrix to square.
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.


    Returns
    -------
    output : ndarray
        Returns squared ndarray.
    """

    if out is None:
        out = empty_like(data)

    out[:] = data ** 2

    return out


def checksum_matrix(
    name,
    raw_matrix,
    size,
    slice_size=16,
    scale=0.0000001,
    ):
    """Checksum matrix and log values

    Parameters
    ----------
    name : str
        Human friendly name used for log header.
    raw_matrix : ndarray
        Matrix or array to checksum.
    size : int
        Number of elements in marix to checksum.
    slice_size : int
        Number of individual elements to show.
    scale : float
        Scaling factor used in checksummming.

    Returns
    -------
    output : list of str
        Returns a list of checksum log entries suitable for printing or
        logging.
    """

    matrix = raw_matrix.ravel()
    output = []
    output.append('=== checksum %s ===' % name)
    output.append('first slice elements:')
    line_elems = 6
    for i in range(slice_size / line_elems):
        slice_view = matrix[i * line_elems:(i + 1) * line_elems]
        output.append(' %s' % ' '.join(['%f' % j for j in slice_view]))
    output.append('mid slice elements:')
    for i in range(slice_size / line_elems):
        first = (size - slice_size) / 2 + i * line_elems
        last = first + line_elems
        slice_view = matrix[first:last]
        output.append(' %s' % ' '.join(['%f' % j for j in slice_view]))
    output.append('last elements:')
    for i in range(slice_size / line_elems):
        first = i * line_elems + size - slice_size
        last = first + line_elems
        slice_view = matrix[first:last]
        output.append(' %s' % ' '.join(['%f' % j for j in slice_view]))
    output.append('sum of first quarter chunk %f' % matrix[:size
                  / 4].sum())
    output.append('sum of second quarter chunk %f' % matrix[size / 4:2
                  * size / 4].sum())
    output.append('sum of third quarter chunk %f' % matrix[2 * size
                  / 4:3 * size / 4].sum())
    output.append('sum of fourth quarter chunk %f' % matrix[3 * size
                  / 4:].sum())
    output.append('sum of center quarter chunk %f' % matrix[size / 2
                  - size / line_elems:size / 2 + size
                  / line_elems].sum())
    output.append('sum of %d elems %f' % (size, matrix.sum()))
    (check_first, check_last) = (0, size)
    native_check = (1. + scale * arange(check_first, check_last,
                    dtype=matrix.dtype)) \
        * (matrix[check_first:check_last] + scale)
    native_sum = native_check.sum()
    output.append('checksum of %d elems is %f' % (size, native_sum))
    return output


def log_checksum(
    title,
    matrix,
    size,
    slice_size=16,
    ):
    """Wrap output from checksum_matrix in log

    Parameters
    ----------
    title : str
        Title string to include in log.
    matrix : ndarray
        An array to checksum.
    size : int
        Number of elements in matrix to checksum.
    slice_size : int
        Number of individual elements to show.
    """

    for entry in checksum_matrix(title, matrix, size, slice_size):
        logging.info(entry)


def check_norm(zero_norm, air_norm):
    """Validate zero norm projection against air norm

    Parameters
    ----------
    zero_norm : ndarray
        Matrix with background intensities.
    air_norm : ndarray
        Matrix with pure air intensities.
    conf : dict
        Configuration dictionary.

    Raises
    ------
    ValueError
        If zero norm is greater than air norm.
    """

    if zero_norm is not None and air_norm is not None and (air_norm
            < zero_norm).any():
        msg = '%s %s' % ('Some zero norm values are greater than ',
                         'their corresponding air norm values')
        raise ValueError(msg)


def interpolate_proj_pixels(
    data,
    proj_mask,
    detector_shape,
    out=None,
    ):
    """Interpolate pixels in *data* based on the pixels marked in *proj_mask*
    Pixels marked for interpolation are set to the average value of it's 
    two neighbours in the row direction. Neighbour values within the mask not 
    taken into account.

    Parameters
    ----------
    data : ndarray
        Data matrix with measured intensities.
    proj_mask : ndarray
        Matrix with pixels marked for interpolation
    proj_scale : ndarray
        Matrix with scaling values for interpolated pixels
    detector_shape : tuple
        Detector shape as tuple with number of rows and columns.
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.
        
    Returns
    -------
    output : ndarray
        Returns ndarray with interpolated projections.
    """

    # Allocate out if not provided

    if out is None:
        out = empty_like(data)

    # Fake stack of projections even for a single flat projection

    orig_data_shape = data.shape
    if data.shape == detector_shape:
        data.shape = tuple([1] + list(data.shape))

    orig_out_shape = out.shape
    if out.shape == detector_shape:
        out.shape = tuple([1] + list(out.shape))

    mask_scale = zeros(detector_shape[1])

    for proj_idx in xrange(data.shape[0]):

        # Copy data to out and extract raw projection
        # If 'data' is 'out' numpy omits the actual copy

        out[proj_idx] = data[proj_idx]
        raw_proj = out[proj_idx]

        for row in xrange(detector_shape[0]):
            mask_idx = nonzero(proj_mask[row])[0]

            if row == 0:
                raw_proj[row, mask_idx] = 0
                mask_scale[mask_idx] = 0
            else:
                prev_mask = ~proj_mask[row - 1, mask_idx]
                raw_proj[row, mask_idx] = raw_proj[row - 1, mask_idx] \
                    * prev_mask
                mask_scale[mask_idx] = prev_mask

            if row < raw_proj.shape[0] - 1:
                next_mask = ~proj_mask[row + 1, mask_idx]
                raw_proj[row, mask_idx] += raw_proj[row + 1, mask_idx] \
                    * next_mask
                mask_scale[mask_idx] += next_mask

            raw_proj[row, mask_idx] /= mask_scale[mask_idx]

    data.shape = orig_data_shape
    out.shape = orig_out_shape

    return out


def generate_proj_filter(
    name,
    width,
    scale,
    nyquist_fraction,
    fdt,
    ):
    """
    Generate projection filter array used 
    for filtering projections prior to reconstruction.

    Parameters
    ----------
    name : str
        Filter name.
    width : int
        Filter width in pixels
    scale : float
        Filter scale factor, if 1.0 filter is not scaled
    nyquist_fraction : float
        http://en.wikipedia.org/wiki/Nyquist_frequency
    fdt : dtype
        Output filter data type.

    Returns
    -------
    output : ndarray
        Returns an filter array of length *width* with dtype *fdt*.
        
    Raises
    ------
    ValueError
        If value of *name* is not a valid filter
    """

    Omega = 1 / scale
    domega = Omega / width

    omega = domega * arange(0, width / 2 + 1, dtype=fdt)
    new_filter = omega

    filter_tail = new_filter[1:]
    omega_tail = omega[1:]
    crop_at_nyquist = True

    if name == 'ram-lak':
        pass
    elif name == 'shepp-logan':
        filter_tail *= sin(pi * omega_tail / (nyquist_fraction
                           * Omega)) / (pi * omega_tail
                / (nyquist_fraction * Omega))
    elif name == 'cosine':
        filter_tail *= cos(pi * omega_tail / (nyquist_fraction * Omega))
    elif name == 'hamming':
        filter_tail *= 0.54 + 0.46 * cos(2 * pi * omega_tail
                / (nyquist_fraction * Omega))
    elif name == 'hann':
        filter_tail *= 0.5 * (1 + cos(2 * pi * omega_tail
                              / (nyquist_fraction * Omega)))
    else:
        raise ValueError('Unsupported filter: \'%s\'' % name)

    if crop_at_nyquist:

        # Crop frequencies at nyquist_fraction * Nyquist freq

        new_filter[omega > 0.5 * Omega * nyquist_fraction] = 0

    # Frequencies ordered according to:
    # http://docs.scipy.org/doc/numpy/reference/routines.fft.html#implementation-details
    #
    # filter[0] contains the zero-frequency term (the mean of the signal)
    # filter[1:n/2] contains the positive-frequency terms
    # filter[n/2+1:] contains the negative-frequency terms
    # filter[n/2] represents both positive and negative Nyquist frequency

    return hstack((new_filter, new_filter[1:-1][::-1]))


def hounsfield_scale(data, raw_voxel_water, out=None):
    """Convert reconstructed data to the hounsfield scale 
    based on the raw voxel value of distilled water.

    Parameters
    ----------
    data : ndarray
        Data matrix with reconstructed raw voxel values
    raw_voxel_water : float
        The raw voxel value of distilled water
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.

    Returns
    -------
    output : ndarray
        Returns ndarray scaled to hounsfield units (HU)
    """

    if out is None:
        out = empty_like(data)

    out[:] = data[:]

    # Hounsfield = 1000 * ((data - raw_voxel_water) / raw_voxel_water)

    out -= raw_voxel_water
    out /= raw_voxel_water
    out *= 1000

    return out


