#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - numpy specific katsevich reconstruction kernels
# Copyright (C) 2011-2013  The CT-Toolbox Project lead by Brian Vinter
#
# This file is part of CT-Toolbox.
#
# CT-Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# CT-Toolbox is distributed in the hope that it will be useful,
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

"""Spiral cone beam CT kernels using the Katsevich algorithm"""

import time

from cphct.log import logging

# These are basic numpy functions exposed through npycore to use same numpy

from cphct.npycore import zeros, zeros_like, arange, int32, sin, cos, \
    arctan, sqrt, interp, convolve, clip, floor, ceil
from cphct.npycore.io import save_auto, get_npy_data


def flat_diff_chunk_vector(
    first,
    last,
    input_array,
    output_array,
    conf,
    ):
    """Run differentiation step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rows, columns) form.

    Parameters
    ----------
    first : int
        First projection to differentiate.
    last : int
        Last projection to differentiate.
    input_array : ndarray
        Input array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Differentiation time.
    """

    before_diff = time.time()
    dia = conf['scan_diameter']
    dia_sqr = dia ** 2
    delta_s = conf['delta_s']
    pixel_span = conf['detector_pixel_span']
    pixel_height = conf['detector_pixel_height']

    # Skip extension and last coordinate for diff

    col_coords = get_npy_data(conf, 'col_coords_ext')[:-2]
    row_coords = get_npy_data(conf, 'row_coords_ext')[:-2]

    # TODO: these helpers can be memory and calculation optimized

    row_col_prod = zeros_like(input_array[0, :-1, :-1])
    row_col_prod += col_coords
    row_transposed = zeros_like(row_coords)
    row_transposed += row_coords
    row_transposed.shape = (len(row_coords), 1)
    row_col_prod *= row_transposed
    col_sqr = zeros_like(input_array[0, :-1, :-1])
    col_sqr += col_coords ** 2
    row_sqr = zeros_like(input_array[0, :-1, :-1])
    row_sqr += row_transposed ** 2

    # we skip last projection and detector pixel due to use of adjacent
    # elements in all directions and no padding

    for proj_index in xrange(first, last):
        proj = proj_index - first

        # Differentiation with respect to projections, rows and columns.
        # Expects input to have that order of dimensions!
        # Use the chain rule with neighboring pixels on adjacent projections

        d_proj = (input_array[proj + 1, :-1, :-1] - input_array[proj, :
                  -1, :-1] + input_array[proj + 1, 1:, :-1]
                  - input_array[proj, 1:, :-1] + input_array[proj + 1, :
                  -1, 1:] - input_array[proj, :-1, 1:]
                  + input_array[proj + 1, 1:, 1:] - input_array[proj, 1:
                  , 1:]) / (4 * delta_s)
        d_row = (input_array[proj, 1:, :-1] - input_array[proj, :-1, :
                 -1] + input_array[proj, 1:, 1:] - input_array[proj, :
                 -1, 1:] + input_array[proj + 1, 1:, :-1]
                 - input_array[proj + 1, :-1, :-1] + input_array[proj
                 + 1, 1:, 1:] - input_array[proj + 1, :-1, 1:]) / (4
                * pixel_height)
        d_col = (input_array[proj, :-1, 1:] - input_array[proj, :-1, :
                 -1] + input_array[proj, 1:, 1:] - input_array[proj, 1:
                 , :-1] + input_array[proj + 1, :-1, 1:]
                 - input_array[proj + 1, :-1, :-1] + input_array[proj
                 + 1, 1:, 1:] - input_array[proj + 1, 1:, :-1]) / (4
                * pixel_span)
        output_array[proj, :-1, :-1] = d_proj + d_col * (col_sqr
                + dia_sqr) / dia + d_row * row_col_prod / dia

        # In-place length correction because detector is flat

        output_array[proj, :-1, :-1] *= dia / sqrt(col_sqr + dia_sqr
                + row_sqr)
    after_diff = time.time()
    diff_time = after_diff - before_diff
    return diff_time


def flat_fwd_rebin_chunk_orig(
    first,
    last,
    input_array,
    output_array,
    conf,
    ):
    """Run forward rebinning step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rows, columns) form.

    Parameters
    ----------
    first : int
        First projection to forward rebin.
    last : int
        Last projection to forward rebin.
    input_array : ndarray
        Input array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Forward rebinning time.
    """

    before_rebin = time.time()
    pixel_height = conf['detector_pixel_height']
    detector_columns = conf['detector_columns']
    fwd_rebin_row = get_npy_data(conf, 'fwd_rebin_row')
    row_coords = get_npy_data(conf, 'row_coords_ext')[:-1]
    for proj_index in xrange(first, last):
        proj = proj_index - first
        for col in xrange(detector_columns):
            rebin_scaled = fwd_rebin_row[:, col] / pixel_height

            # Map scaled coordinates into original row index range of integers
            # for rebinning

            output_array[proj, :, col] = interp(rebin_scaled,
                    row_coords, input_array[proj, :, col])
    after_rebin = time.time()
    rebin_time = after_rebin - before_rebin
    return rebin_time


def flat_fwd_rebin_chunk_single(
    first,
    last,
    input_array,
    output_array,
    conf,
    ):
    """Run forward rebinning step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rows, columns) form.

    Parameters
    ----------
    first : int
        First projection to forward rebin.
    last : int
        Last projection to forward rebin.
    input_array : ndarray
        Input array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Forward rebinning time.
    """

    before_rebin = time.time()
    pixel_height = conf['detector_pixel_height']
    detector_rows = conf['detector_rows']
    detector_columns = conf['detector_columns']
    detector_rebin_rows = conf['detector_rebin_rows']
    detector_row_offset = conf['detector_row_offset']
    fwd_rebin_row = get_npy_data(conf, 'fwd_rebin_row')
    for proj_index in xrange(first, last):
        proj = proj_index - first
        for rebin_row in xrange(detector_rebin_rows):
            for col in xrange(detector_columns):
                fwd_remap = fwd_rebin_row[rebin_row, col]

                # Translate row coordinate to row index in projection
                # Sign is inverted for shift and thus also for offset.

                # TODO: rows-1 here too?

                row_scaled = fwd_remap / pixel_height + 0.5 \
                    * detector_rows - detector_row_offset

                # make sure row and row+1 are in valid row range

                row_scaled = min(max(0, row_scaled), detector_rows - 2)
                row = int(row_scaled)
                row_frac = row_scaled - row
                val = (1 - row_frac) * input_array[proj, row, col] \
                    + row_frac * input_array[proj, row + 1, col]
                output_array[proj, rebin_row, col] = val
    after_rebin = time.time()
    rebin_time = after_rebin - before_rebin
    return rebin_time


def flat_fwd_rebin_chunk_vector(
    first,
    last,
    input_array,
    output_array,
    conf,
    ):
    """Run forward rebinning step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rows, columns) form.

    Parameters
    ----------
    first : int
        First projection to forward rebin.
    last : int
        Last projection to forward rebin.
    input_array : ndarray
        Input array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Forward rebinning time.
    """

    before_rebin = time.time()
    pixel_height = conf['detector_pixel_height']
    detector_rows = conf['detector_rows']
    detector_columns = conf['detector_columns']
    detector_row_offset = conf['detector_row_offset']
    fwd_rebin_row = get_npy_data(conf, 'fwd_rebin_row')
    for proj_index in xrange(first, last):
        proj = proj_index - first
        for col in xrange(detector_columns):

            # Map scaled coordinates into original row index range of integers
            # for rebinning
            # Sign is inverted for shift and thus also for offset.

            # TODO: rows-1 here too?

            row_scaled = fwd_rebin_row[:, col] / pixel_height + 0.5 \
                * detector_rows - detector_row_offset

            # make sure row and row+1 are in valid row range

            clip(row_scaled, 0, detector_rows - 2, row_scaled)

            # we need integer indexes (this repeated creation may be slow)

            row = floor(row_scaled).astype(int32)

            # linear interpolation of row neighbors

            row_frac = row_scaled - row
            output_array[proj, :, col] = (1 - row_frac) \
                * input_array[proj, row, col] + row_frac \
                * input_array[proj, row + 1, col]
    after_rebin = time.time()
    rebin_time = after_rebin - before_rebin
    return rebin_time


def flat_conv_chunk_vector(
    first,
    last,
    input_array,
    hilbert_array,
    output_array,
    conf,
    ):
    """Run convolution step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rebin, columns) form.

    Parameters
    ----------
    first : int
        First projection to convolve.
    last : int
        Last projection to convolve.
    input_array : ndarray
        Input array.
    hilbert_array : ndarray
        Hilbert convolution helper array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Convolution time.
    """

    before_conv = time.time()
    detector_columns = conf['detector_columns']
    detector_rebin_rows = conf['detector_rebin_rows']

    # TODO: use rectangular hilbert window as suggested in Noo paper?

    for proj_index in xrange(first, last):
        proj = proj_index - first
        for rebin_row in xrange(detector_rebin_rows):

            # use convolve for now instead of manual convolution sum
            # yields len(hilbert_array)+detector_columns-1 elements

            filter_conv = convolve(hilbert_array, input_array[proj,
                                   rebin_row, :])

            # only use central elements of convolution

            tmp = filter_conv[detector_columns - 1:2 * detector_columns
                - 1]
            output_array[proj, rebin_row, :] = tmp
    after_conv = time.time()
    conv_time = after_conv - before_conv
    return conv_time


def flat_rev_rebin_chunk_single(
    first,
    last,
    input_array,
    output_array,
    conf,
    ):
    """Run reverse rebinning step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rebin, columns) form.

    Parameters
    ----------
    first : int
        First projection to reverse rebin.
    last : int
        Last projection to reverse rebin.
    input_array : ndarray
        Input array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Reverse rebinning time.
    """

    detector_rows = conf['detector_rows']
    detector_columns = conf['detector_columns']
    detector_column_offset = conf['detector_column_offset']
    detector_rebin_rows = conf['detector_rebin_rows']
    detector_column_offset = conf['detector_column_offset']
    row_coords = get_npy_data(conf, 'row_coords_ext')[:-1]
    fwd_rebin_row = get_npy_data(conf, 'fwd_rebin_row')
    (src, dst) = (input_array, output_array)
    before_rebin = time.time()
    fracs = zeros(2, dtype=conf['data_type'])

    # col offset ruins positive/negative half split

    # TODO: columns-1 here too?

    pos_start = int(0.5 * detector_columns - detector_column_offset)
    for proj_index in xrange(first, last):
        proj = proj_index - first
        for row in xrange(detector_rows):

            # column coordinate in positive range

            for col in xrange(pos_start, detector_columns):

                # Find the rebin row index that fits limits (zero as default)

                rebin_row = 0
                rebin_col = col
                (fracs[0], fracs[1]) = (0.0, 1.0)
                for rebin in xrange(detector_rebin_rows - 1):
                    if row_coords[row] >= fwd_rebin_row[rebin, col] \
                        and row_coords[row] <= fwd_rebin_row[rebin + 1,
                            col]:
                        rebin_row = rebin
                        fracs[0] = (row_coords[row]
                                    - fwd_rebin_row[rebin_row,
                                    rebin_col]) \
                            / (fwd_rebin_row[rebin_row + 1, rebin_col]
                               - fwd_rebin_row[rebin_row, rebin_col])
                        fracs[1] -= fracs[0]
                        break
                dst[proj, row, col] = fracs[1] * src[proj, rebin_row,
                        rebin_col] + fracs[0] * src[proj, rebin_row
                        + 1, rebin_col]

            # column coordinate in negative range

            for col in xrange(pos_start):

                # Find the rebin row index that fits limits (one as default)

                rebin_row = 1
                rebin_col = col
                (fracs[0], fracs[1]) = (0.0, 1.0)
                for rebin in xrange(detector_rebin_rows - 1, 0, -1):
                    if row_coords[row] >= fwd_rebin_row[rebin - 1, col] \
                        and row_coords[row] <= fwd_rebin_row[rebin,
                            col]:
                        rebin_row = rebin
                        rebin_col = col
                        fracs[0] = (row_coords[row]
                                    - fwd_rebin_row[rebin_row - 1,
                                    rebin_col]) \
                            / (fwd_rebin_row[rebin_row, rebin_col]
                               - fwd_rebin_row[rebin_row - 1,
                               rebin_col])
                        fracs[1] -= fracs[0]
                        break
                dst[proj, row, col] = fracs[1] * src[proj, rebin_row
                        - 1, rebin_col] + fracs[0] * src[proj,
                        rebin_row, rebin_col]
    after_rebin = time.time()
    rebin_time = after_rebin - before_rebin
    return rebin_time


def curved_diff_chunk_vector(
    first,
    last,
    input_array,
    output_array,
    conf,
    ):
    """Run differentiation step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rows, columns) form.

    Parameters
    ----------
    first : int
        First projection to differentiate.
    last : int
        Last projection to differentiate.
    input_array : ndarray
        Input array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Differentiation time.
    """

    before_diff = time.time()
    dia = conf['scan_diameter']
    dia_sqr = dia ** 2
    delta_s = conf['delta_s']
    pixel_span = conf['detector_pixel_span']

    # Skip extension and last coordinate for diff

    row_coords = get_npy_data(conf, 'row_coords_ext')[:-2]

    # TODO: these helpers can be memory and calculation optimized

    row_transposed = zeros_like(row_coords)
    row_transposed += row_coords
    row_transposed.shape = (len(row_coords), 1)
    row_sqr = zeros_like(input_array[0, :-1, :-1])
    row_sqr += row_transposed ** 2

    # we skip last projection and detector pixel due to use of adjacent
    # elements in all directions and no padding

    for proj_index in xrange(first, last):
        proj = proj_index - first

        # Differentiation with respect to projections, rows and columns.
        # Expects input to have that order of dimensions!
        # Use the chain rule with neighboring pixels on adjacent projections

        d_proj = (input_array[proj + 1, :-1, :-1] - input_array[proj, :
                  -1, :-1] + input_array[proj + 1, :-1, 1:]
                  - input_array[proj, :-1, 1:]) / (2 * delta_s)
        d_col = (input_array[proj, :-1, 1:] - input_array[proj, :-1, :
                 -1] + input_array[proj + 1, :-1, 1:]
                 - input_array[proj + 1, :-1, :-1]) / (2 * pixel_span)
        output_array[proj, :-1, :-1] = d_proj + d_col

        # In-place length correction because detector is flat in one direction

        output_array[proj, :-1, :-1] *= dia / sqrt(dia_sqr + row_sqr)
    after_diff = time.time()
    diff_time = after_diff - before_diff
    return diff_time


def curved_fwd_rebin_chunk_vector(
    first,
    last,
    input_array,
    output_array,
    conf,
    ):
    """Run forward rebinning step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rows, columns) form.

    Parameters
    ----------
    first : int
        First projection to forward rebin.
    last : int
        Last projection to forward rebin.
    input_array : ndarray
        Input array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Forward rebinning time.
    """

    # Identical to flat case

    return flat_fwd_rebin_chunk_vector(first, last, input_array,
            output_array, conf)


def curved_conv_chunk_vector(
    first,
    last,
    input_array,
    hilbert_array,
    output_array,
    conf,
    ):
    """Run convolution step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rebin, columns) form.

    Parameters
    ----------
    first : int
        First projection to convolve.
    last : int
        Last projection to convolve.
    input_array : ndarray
        Input array.
    hilbert_array : ndarray
        Hilbert convolution helper array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Convolution time.
    """

    # Identical to flat case

    return flat_conv_chunk_vector(first, last, input_array, hilbert_array,
                                  output_array, conf)


def curved_rev_rebin_chunk_single(
    first,
    last,
    input_array,
    output_array,
    conf,
    ):
    """Run reverse rebinning step on chunk of projections keeping the result in
    output_array.
    Expects input_array to be on (projections, rebin, columns) form.

    Parameters
    ----------
    first : int
        First projection to reverse rebin.
    last : int
        Last projection to reverse rebin.
    input_array : ndarray
        Input array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Reverse rebinning time.
    """

    col_coords = get_npy_data(conf, 'col_coords_ext')[:-1]

    # Identical to flat case but with cosinus weighting afterwards

    before_rebin = time.time()
    rebin_time = flat_rev_rebin_chunk_single(first, last, input_array,
            output_array, conf)

    # multiply each column element with cosinus to that column coordinate

    output_array *= cos(col_coords)
    after_rebin = time.time()
    rebin_time = after_rebin - before_rebin
    return rebin_time


def flat_backproject_chunk(
    chunk_index,
    first_proj,
    last_proj,
    first_z,
    last_z,
    input_array,
    row_mins_array,
    row_maxs_array,
    output_array,
    conf,
    ):
    """Run backprojection on chunk of projections keeping the results in
    output_array.

    Parameters
    ----------
    chunk_index : int
        Index of chunk in chunked backprojection.
    first_proj : int
        Index of first projection to include in chunked backprojection.
    last_proj : int
        Index of last projection to include in chunked backprojection.
    first_z : int
        Index of first z voxels to include in chunked backprojection.
    last_z : int
        Index of last z voxels to include in chunked backprojection.
    input_array : ndarray
        Input array.
    row_mins_array : ndarray
        Row interpolation helper array.
    row_maxs_array : ndarray
        Row interpolation helper array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Backprojection time.
    """

    before_chunk = time.time()

    # Limit to actual projection sources in chunk

    source_pos = get_npy_data(conf, 'source_pos')[first_proj:last_proj + 1]
    scan_radius = conf['scan_radius']
    scan_diameter = conf['scan_diameter']
    x_min = conf['x_min']
    y_min = conf['y_min']
    z_min = conf['z_min']
    delta_x = conf['delta_x']
    delta_y = conf['delta_y']
    delta_z = conf['delta_z']
    x_voxels = conf['x_voxels']
    y_voxels = conf['y_voxels']
    z_voxels = conf['z_voxels']
    fov_radius = conf['fov_radius']
    pixel_span = conf['detector_pixel_span']
    pixel_height = conf['detector_pixel_height']
    detector_rows = conf['detector_rows']
    detector_columns = conf['detector_columns']
    detector_row_offset = conf['detector_row_offset']
    detector_column_offset = conf['detector_column_offset']
    detector_row_shift = conf['detector_row_shift']
    progress_per_radian = conf['progress_per_radian']

    (prev_proj, cur_proj, next_proj) = range(3)

    # Calculate x, y and squared coordinates once and for all

    x_coords = arange(x_voxels, dtype=conf['data_type']) * delta_x \
        + x_min
    sqr_x_coords = x_coords ** 2
    y_coords = arange(y_voxels, dtype=conf['data_type']) * delta_y \
        + y_min
    sqr_y_coords = y_coords ** 2
    rad_sqr = fov_radius ** 2
    proj_row_coords = zeros(3, dtype=conf['data_type'])
    proj_row_steps = zeros(3, dtype=conf['data_type'])
    (debug_x, debug_y) = (x_voxels / 2, y_voxels / 2)
    for x in xrange(x_voxels):
        x_coord = x_coords[x]
        x_coord_sqr = sqr_x_coords[x]
        for y in xrange(y_voxels):
            y_coord = y_coords[y]
            y_coord_sqr = sqr_y_coords[y]

            # Ignore voxels with center outside the cylinder with fov_radius

            if x_coord_sqr + y_coord_sqr > rad_sqr:
                continue

            if y == debug_y:
                logging.debug('back project (%d, %d, %d:%d) with projs %d:%d'
                               % (
                    x,
                    y,
                    first_z,
                    last_z,
                    first_proj,
                    last_proj,
                    ))

            # Constant helper arrays for this particular (x,y) and all angles.
            # Column and scale helpers remain constant for all z-values but
            # row helpers must be calculated for each z.

            # The 'denominator' or scaling function

            scale_helpers = scan_radius - x_coord * cos(source_pos) \
                - y_coord * sin(source_pos)

            # Projected (float) coordinates from column projection formula

            proj_col_coords = scan_diameter * (-x_coord
                    * sin(source_pos) + y_coord * cos(source_pos)) \
                / scale_helpers

            # Matching column indices in exact (float) and whole (integer) form
            # We divide signed coordinate by size and shift by half the number
            # of pixels to get unsigned pixel index when rounding to integer.
            # We need to either round towards zero or limit range to actual
            # index range to avoid hits exactly on the borders to result in out
            # of bounds index.
            # Sign is inverted for shift and thus also for offset.

            # TODO: columns-1 here too?

            proj_col_reals = proj_col_coords / pixel_span + 0.5 \
                * detector_columns - detector_column_offset
            proj_col_ints = proj_col_reals.astype(int32)

            # TODO: clip to det-2 here? (we do that later anyway)

            clip(proj_col_ints, 0, detector_columns, proj_col_ints)
            proj_col_fracs = proj_col_reals - proj_col_ints

            # Row coordinate step for each z increment: this equals the
            # derivative with respect to z of the row projection formula

            proj_row_coord_diffs = scan_diameter / scale_helpers

            # Row index step for each z index increment: same as above but
            # scaled to be in z index instead of coordinate

            proj_row_ind_diffs = proj_row_coord_diffs * delta_z \
                / pixel_height

            # Row coordinates for z_min using row coordinate formula.
            # Used to calculate the row index for any z index in the z loop

            proj_row_coord_z_min = scan_diameter * (z_min
                    - progress_per_radian * source_pos) / scale_helpers
            proj_row_ind_z_min = proj_row_coord_z_min / pixel_height \
                + detector_row_shift

            # Interpolate nearest precalculated neighbors in limit row coords.
            # They are used as row coordinate boundaries for z loop and in
            # boundary weigths.
            # Please note that row_mins/maxs are built from the extended
            # col coords so that they include one extra element to allow this
            # interpolation even for the last valid column index,
            # (detector_columns-1)

            proj_row_coord_mins = (1 - proj_col_fracs) \
                * row_mins_array[proj_col_ints] + proj_col_fracs \
                * row_mins_array[proj_col_ints + 1]
            proj_row_coord_maxs = (1 - proj_col_fracs) \
                * row_maxs_array[proj_col_ints] + proj_col_fracs \
                * row_maxs_array[proj_col_ints + 1]

            # Use row projection formula to calculate z limits from row limits

            z_coord_mins = source_pos * progress_per_radian \
                + proj_row_coord_mins * scale_helpers / scan_diameter
            z_coord_maxs = source_pos * progress_per_radian \
                + proj_row_coord_maxs * scale_helpers / scan_diameter

            # Extract naive integer indices - handle out of bounds later
            # We round inwards using ceil and floor respectively to avoid
            # excess contribution from border pixels

            z_firsts = ceil((z_coord_mins - z_min)
                            / delta_z).astype(int32)
            z_lasts = floor((z_coord_maxs - z_min)
                            / delta_z).astype(int32)
            for proj_index in xrange(first_proj, last_proj):
                proj = proj_index - first_proj

                # Reset proj_row_coords triple to first row coordinates before
                # each z loop.
                # Please note that proj_row_coords prev and next values are
                # *only* used for projs where prev and next makes sense below.
                # So we just ignore the values for out of bounds border cases.

                proj_row_coords[:] = 0
                proj_row_steps[:] = 0
                if proj > 0:
                    proj_row_steps[prev_proj] = \
                        proj_row_coord_diffs[proj - 1] * delta_z
                    proj_row_coords[prev_proj] = \
                        proj_row_coord_z_min[proj - 1] + z_firsts[proj] \
                        * proj_row_steps[prev_proj]
                proj_row_steps[cur_proj] = proj_row_coord_diffs[proj] \
                    * delta_z
                proj_row_coords[cur_proj] = proj_row_coord_z_min[proj] \
                    + z_firsts[proj] * proj_row_steps[cur_proj]
                if proj < last_proj - 1:
                    proj_row_steps[next_proj] = \
                        proj_row_coord_diffs[proj + 1] * delta_z
                    proj_row_coords[next_proj] = \
                        proj_row_coord_z_min[proj + 1] + z_firsts[proj] \
                        * proj_row_steps[next_proj]
                if z_coord_maxs[proj] < z_min + first_z * delta_z \
                    or z_coord_mins[proj] > z_min + last_z * delta_z:
                    continue
                if x == debug_x and y == debug_y:
                    logging.debug('loop (%d, %d, %d:%d) proj %d %f:%f'
                                  % (
                        x,
                        y,
                        z_firsts[proj],
                        z_lasts[proj],
                        proj_index,
                        proj_row_coord_mins[proj] / pixel_height + 
                                      detector_row_shift,
                        proj_row_coord_maxs[proj] / pixel_height +
                                      detector_row_shift,
                        ))

                # Include last z index

                for z in xrange(z_firsts[proj], z_lasts[proj] + 1):

                    # Always update projected row coordinates

                    (prev_row_coord, cur_row_coord, next_row_coord) = \
                        proj_row_coords[:]
                    proj_row_coords += proj_row_steps

                    # Skip out of bounds indices here to avoid border weighting
                    # on boundary clipped index values

                    if z < first_z or z > last_z:
                        continue
                    z_coord = z_min + z * delta_z
                    z_local = z - first_z

                    # Border weight only applies for first and last z

                    if z == z_firsts[proj] and next_row_coord \
                        < proj_row_coord_mins[proj + 1]:
                        weight = 0.5 + (z_coord - z_coord_mins[proj]) \
                            / (z_coord_mins[proj + 1]
                               - z_coord_mins[proj])
                        if x == debug_x and y == debug_y:
                            logging.debug('first weight: %f %f %f %f %f'
                                     % (next_row_coord,
                                    proj_row_coord_mins[proj + 1],
                                    z_coord_mins[proj + 1], z_coord,
                                    z_coord_mins[proj]))
                    elif z == z_lasts[proj] and prev_row_coord \
                        > proj_row_coord_maxs[proj - 1]:
                        weight = 0.5 + (z_coord_maxs[proj] - z_coord) \
                            / (z_coord_maxs[proj] - z_coord_maxs[proj
                               - 1])
                        if x == debug_x and y == debug_y:
                            logging.debug('last weight: %f %f %f %f %f'
                                    % (prev_row_coord,
                                    proj_row_coord_mins[proj - 1],
                                    z_coord_maxs[proj], z_coord,
                                    z_coord_maxs[proj - 1]))
                    else:
                        weight = 1.0

                    # TODO: is this correct? (0.5 less than direct proj coord)
                    # ... obviously from the (detector_rows-1) in
                    # proj_row_ind_z_min
                    # Removing -1 breaks result, inspect same -1 for col?
                    # Row indices in exact (real) and whole (integer) form.
                    # Offset is already included in proj_row_ind_z_min

                    proj_row_real = proj_row_ind_z_min[proj] + z \
                        * proj_row_ind_diffs[proj]
                    proj_row_int = proj_row_real.astype(int32)

                    # make sure row and row+1 are in valid row range

                    proj_row_int = min(max(proj_row_int, 0),
                            detector_rows - 2)
                    proj_row_frac = proj_row_real - proj_row_int
                    proj_row_int_next = proj_row_int + 1
                    proj_col_int = proj_col_ints[proj]

                    # make sure col and col+1 are in valid col range

                    proj_col_int = min(max(proj_col_int, 0),
                            detector_columns - 2)
                    proj_col_frac = proj_col_fracs[proj]
                    proj_col_int_next = proj_col_int + 1
                    proj_mean = input_array[proj, proj_row_int,
                            proj_col_int] * (1 - proj_row_frac) * (1
                            - proj_col_frac) + input_array[proj,
                            proj_row_int_next, proj_col_int] \
                        * proj_row_frac * (1 - proj_col_frac) \
                        + input_array[proj, proj_row_int,
                            proj_col_int_next] * (1 - proj_row_frac) \
                        * proj_col_frac + input_array[proj,
                            proj_row_int_next, proj_col_int_next] \
                        * proj_row_frac * proj_col_frac
                    contrib = weight / scale_helpers[proj] * proj_mean
                    output_array[x, y, z_local] += contrib
                    if x == debug_x and y == debug_y:
                        logging.debug('update (%d, %d, %d): %f (%f) from %d'
                                 % (
                            x,
                            y,
                            z,
                            contrib,
                            output_array[x, y, z_local],
                            proj_index,
                            ))
                        logging.debug('w %f r %d %f (%f) c %d %f (%f) m %f'
                                 % (
                            weight,
                            proj_row_int,
                            proj_row_frac,
                            proj_row_real,
                            proj_col_int,
                            proj_col_frac,
                            proj_col_reals[proj],
                            proj_mean,
                            ))
    after_chunk = time.time()
    chunk_time = after_chunk - before_chunk
    logging.debug('finished backproject kernel in %ss' % chunk_time)

    # Actually we should scale result with delta_s/(2*pi) = projs_per_turn here
    # But we delay that until after the function is called

    return chunk_time


def curved_backproject_chunk(
    chunk_index,
    first_proj,
    last_proj,
    first_z,
    last_z,
    input_array,
    row_mins_array,
    row_maxs_array,
    output_array,
    conf,
    ):
    """Run backprojection on chunk of projections keeping the results in
    output_array.

    Parameters
    ----------
    chunk_index : int
        Index of chunk in chunked backprojection.
    first_proj : int
        Index of first projection to include in chunked backprojection.
    last_proj : int
        Index of last projection to include in chunked backprojection.
    first_z : int
        Index of first z voxels to include in chunked backprojection.
    last_z : int
        Index of last z voxels to include in chunked backprojection.
    input_array : ndarray
        Input array.
    row_mins_array : ndarray
        Row interpolation helper array.
    row_maxs_array : ndarray
        Row interpolation helper array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Backprojection time.
    """

    before_chunk = time.time()

    # Limit to actual projection sources in chunk

    source_pos = get_npy_data(conf, 'source_pos')[first_proj:last_proj + 1]
    scan_radius = conf['scan_radius']
    scan_diameter = conf['scan_diameter']
    x_min = conf['x_min']
    y_min = conf['y_min']
    z_min = conf['z_min']
    delta_x = conf['delta_x']
    delta_y = conf['delta_y']
    delta_z = conf['delta_z']
    x_voxels = conf['x_voxels']
    y_voxels = conf['y_voxels']
    z_voxels = conf['z_voxels']
    fov_radius = conf['fov_radius']
    pixel_span = conf['detector_pixel_span']
    pixel_height = conf['detector_pixel_height']
    detector_rows = conf['detector_rows']
    detector_columns = conf['detector_columns']
    detector_row_offset = conf['detector_row_offset']
    detector_column_offset = conf['detector_column_offset']
    detector_row_shift = conf['detector_row_shift']
    progress_per_radian = conf['progress_per_radian']

    (prev_proj, cur_proj, next_proj) = range(3)

    # Calculate x, y and squared coordinates once and for all

    x_coords = arange(x_voxels, dtype=conf['data_type']) * delta_x \
        + x_min
    sqr_x_coords = x_coords ** 2
    y_coords = arange(y_voxels, dtype=conf['data_type']) * delta_y \
        + y_min
    sqr_y_coords = y_coords ** 2
    rad_sqr = fov_radius ** 2
    proj_row_coords = zeros(3, dtype=conf['data_type'])
    proj_row_steps = zeros(3, dtype=conf['data_type'])
    (debug_x, debug_y) = (x_voxels / 2, y_voxels / 2)
    for x in xrange(x_voxels):
        x_coord = x_coords[x]
        x_coord_sqr = sqr_x_coords[x]
        for y in xrange(y_voxels):
            y_coord = y_coords[y]
            y_coord_sqr = sqr_y_coords[y]

            # Ignore voxels with center outside the cylinder with fov_radius

            if x_coord_sqr + y_coord_sqr > rad_sqr:
                continue

            if y == debug_y:
                logging.debug('back project (%d, %d, %d:%d) with projs %d:%d'
                               % (
                    x,
                    y,
                    first_z,
                    last_z,
                    first_proj,
                    last_proj,
                    ))

            # Constant helper arrays for this particular (x,y) and all angles.
            # Column and scale helpers remain constant for all z-values but
            # row helpers must be calculated for each z.

            # The 'denominator' or scaling function

            scale_helpers = scan_radius - x_coord * cos(source_pos) \
                - y_coord * sin(source_pos)

            # Projected (float) coordinates from column projection formula

            proj_col_coords = arctan((-x_coord * sin(source_pos)
                    + y_coord * cos(source_pos)) / scale_helpers)

            # Avoid repeated calculation of cosinus

            cos_proj_col_coords = cos(proj_col_coords)

            # Matching column indices in exact (float) and whole (integer) form
            # We divide signed coordinate by size and shift by half the number
            # of pixels to get unsigned pixel index when rounding to integer.
            # We need to either round towards zero or limit range to actual
            # index range to avoid hits exactly on the borders to result in out
            # of bounds index.
            # Sign is inverted for shift and thus also for offset.

            # TODO: columns-1 here too?

            proj_col_reals = proj_col_coords / pixel_span + 0.5 \
                * detector_columns - detector_column_offset
            proj_col_ints = proj_col_reals.astype(int32)

            # TODO: clip to det-2 here? (we do that later anyway)

            clip(proj_col_ints, 0, detector_columns, proj_col_ints)
            proj_col_fracs = proj_col_reals - proj_col_ints

            # Row coordinate step for each z increment: this equals the
            # derivative with respect to z of the row projection formula

            proj_row_coord_diffs = scan_diameter * cos_proj_col_coords \
                / scale_helpers

            # Row index step for each z index increment: same as above but
            # scaled to be in z index instead of coordinate

            proj_row_ind_diffs = proj_row_coord_diffs * delta_z \
                / pixel_height

            # Row coordinates for z_min using row coordinate formula.
            # Used to calculate the row index for any z index in the z loop

            proj_row_coord_z_min = scan_diameter * cos_proj_col_coords \
                * (z_min - progress_per_radian * source_pos) \
                / scale_helpers
            proj_row_ind_z_min = proj_row_coord_z_min / pixel_height \
                + detector_row_shift

            # Interpolate nearest precalculated neighbors in limit row coords.
            # They are used as row coordinate boundaries for z loop and in
            # boundary weigths.
            # Please note that row_mins/maxs are built from the extended
            # col coords so that they include one extra element to allow this
            # interpolation even for the last valid column index,
            # (detector_columns-1)

            proj_row_coord_mins = (1 - proj_col_fracs) \
                * row_mins_array[proj_col_ints] + proj_col_fracs \
                * row_mins_array[proj_col_ints + 1]
            proj_row_coord_maxs = (1 - proj_col_fracs) \
                * row_maxs_array[proj_col_ints] + proj_col_fracs \
                * row_maxs_array[proj_col_ints + 1]

            # Use row projection formula to calculate z limits from row limits

            z_coord_mins = source_pos * progress_per_radian \
                + proj_row_coord_mins * scale_helpers / (scan_diameter
                    * cos_proj_col_coords)
            z_coord_maxs = source_pos * progress_per_radian \
                + proj_row_coord_maxs * scale_helpers / (scan_diameter
                    * cos_proj_col_coords)

            # Extract naive integer indices - handle out of bounds later
            # We round inwards using ceil and floor respectively to avoid
            # excess contribution from border pixels

            z_firsts = ceil((z_coord_mins - z_min)
                            / delta_z).astype(int32)
            z_lasts = floor((z_coord_maxs - z_min)
                            / delta_z).astype(int32)
            for proj_index in xrange(first_proj, last_proj):
                proj = proj_index - first_proj

                # Reset proj_row_coords triple to first row coordinates before
                # each z loop.
                # Please note that proj_row_coords prev and next values are
                # *only* used for projs where prev and next makes sense below.
                # So we just ignore the values for out of bounds border cases.

                proj_row_coords[:] = 0
                proj_row_steps[:] = 0
                if proj > 0:
                    proj_row_steps[prev_proj] = \
                        proj_row_coord_diffs[proj - 1] * delta_z
                    proj_row_coords[prev_proj] = \
                        proj_row_coord_z_min[proj - 1] + z_firsts[proj] \
                        * proj_row_steps[prev_proj]
                proj_row_steps[cur_proj] = proj_row_coord_diffs[proj] \
                    * delta_z
                proj_row_coords[cur_proj] = proj_row_coord_z_min[proj] \
                    + z_firsts[proj] * proj_row_steps[cur_proj]
                if proj < last_proj - 1:
                    proj_row_steps[next_proj] = \
                        proj_row_coord_diffs[proj + 1] * delta_z
                    proj_row_coords[next_proj] = \
                        proj_row_coord_z_min[proj + 1] + z_firsts[proj] \
                        * proj_row_steps[next_proj]
                if z_coord_maxs[proj] < z_min + first_z * delta_z \
                    or z_coord_mins[proj] > z_min + last_z * delta_z:
                    continue
                if x == debug_x and y == debug_y:
                    logging.debug('loop (%d, %d, %d:%d) proj %d %f:%f'
                                  % (
                        x,
                        y,
                        z_firsts[proj],
                        z_lasts[proj],
                        proj_index,
                        proj_row_coord_mins[proj] / pixel_height +
                                      detector_row_shift,
                        proj_row_coord_maxs[proj] / pixel_height +
                                      detector_row_shift,
                        ))

                # Include last z index

                for z in xrange(z_firsts[proj], z_lasts[proj] + 1):

                    # Always update projected row coordinates

                    (prev_row_coord, cur_row_coord, next_row_coord) = \
                        proj_row_coords[:]
                    proj_row_coords += proj_row_steps

                    # Skip out of bounds indices here to avoid border weighting
                    # on boundary clipped index values

                    if z < first_z or z > last_z:
                        continue
                    z_coord = z_min + z * delta_z
                    z_local = z - first_z

                    # Border weight only applies for first and last z

                    if z == z_firsts[proj] and next_row_coord \
                        < proj_row_coord_mins[proj + 1]:
                        weight = 0.5 + (z_coord - z_coord_mins[proj]) \
                            / (z_coord_mins[proj + 1]
                               - z_coord_mins[proj])
                        if x == debug_x and y == debug_y:
                            logging.debug('first weight: %f %f %f %f %f'
                                     % (next_row_coord,
                                    proj_row_coord_mins[proj + 1],
                                    z_coord_mins[proj + 1], z_coord,
                                    z_coord_mins[proj]))
                    elif z == z_lasts[proj] and prev_row_coord \
                        > proj_row_coord_maxs[proj - 1]:
                        weight = 0.5 + (z_coord_maxs[proj] - z_coord) \
                            / (z_coord_maxs[proj] - z_coord_maxs[proj
                               - 1])
                        if x == debug_x and y == debug_y:
                            logging.debug('last weight: %f %f %f %f %f'
                                    % (prev_row_coord,
                                    proj_row_coord_mins[proj - 1],
                                    z_coord_maxs[proj], z_coord,
                                    z_coord_maxs[proj - 1]))
                    else:
                        weight = 1.0

                    # TODO: is this correct? (0.5 less than direct proj coord)
                    # ... obviously from the (detector_rows-1) in
                    # proj_row_ind_z_min
                    # Removing -1 breaks result, inspect same -1 for col?
                    # Row indices in exact (real) and whole (integer) form
                    # Offset is already included in proj_row_ind_z_min.

                    proj_row_real = proj_row_ind_z_min[proj] + z \
                        * proj_row_ind_diffs[proj]
                    proj_row_int = proj_row_real.astype(int32)

                    # make sure row and row+1 are in valid row range

                    proj_row_int = min(max(proj_row_int, 0),
                            detector_rows - 2)
                    proj_row_frac = proj_row_real - proj_row_int
                    proj_row_int_next = proj_row_int + 1
                    proj_col_int = proj_col_ints[proj]

                    # make sure col and col+1 are in valid col range

                    proj_col_int = min(max(proj_col_int, 0),
                            detector_columns - 2)
                    proj_col_frac = proj_col_fracs[proj]
                    proj_col_int_next = proj_col_int + 1
                    proj_mean = input_array[proj, proj_row_int,
                            proj_col_int] * (1 - proj_row_frac) * (1
                            - proj_col_frac) + input_array[proj,
                            proj_row_int_next, proj_col_int] \
                        * proj_row_frac * (1 - proj_col_frac) \
                        + input_array[proj, proj_row_int,
                            proj_col_int_next] * (1 - proj_row_frac) \
                        * proj_col_frac + input_array[proj,
                            proj_row_int_next, proj_col_int_next] \
                        * proj_row_frac * proj_col_frac
                    contrib = weight / scale_helpers[proj] * proj_mean
                    output_array[x, y, z_local] += contrib
                    if x == debug_x and y == debug_y:
                        logging.debug('update (%d, %d, %d): %f (%f) from %d'
                                 % (
                            x,
                            y,
                            z,
                            contrib,
                            output_array[x, y, z_local],
                            proj_index,
                            ))
                        logging.debug('w %f r %d %f (%f) c %d %f (%f) m %f'
                                 % (
                            weight,
                            proj_row_int,
                            proj_row_frac,
                            proj_row_real,
                            proj_col_int,
                            proj_col_frac,
                            proj_col_reals[proj],
                            proj_mean,
                            ))
    after_chunk = time.time()
    chunk_time = after_chunk - before_chunk
    logging.debug('finished backproject kernel in %ss' % chunk_time)

    # Actually we should scale result with delta_s/(2*pi) = projs_per_turn here
    # But we delay that until after the function is called

    return chunk_time


def filter_chunk(
    chunk_index,
    first_proj,
    last_proj,
    input_array,
    diff_array,
    rebin_array,
    hilbert_array,
    conv_array,
    output_array,
    conf,
    ):
    """Run filter on chunk of projections keeping the filtered projections
    in output_array. The first and last argument are projection indices for the
    first and last input projections. The differentiation step uses an extra
    projection, so filtering produces filtered projections with indices from
    first to last-1.

    Parameters
    ----------
    chunk_index : int
        Index of chunk in chunked backprojection.
    first_proj : int
        Index of first projection to include in chunked filtering.
    last_proj : int
        Index of last projection to include in chunked filtering.
    input_array : ndarray
        Input array.
    diff_array : ndarray
        Differentiation helper array.
    rebin_array : ndarray
        Rebinning helper array.
    conv_array : ndarray
        Convolution helper array.
    hilbert_array : ndarray
        Hilbert convolution helper array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Filtering time.
    """

    if conf['detector_shape'] == 'flat':

        # use vectorized forms

        diff_chunk = flat_diff_chunk_vector
        fwd_rebin_chunk = flat_fwd_rebin_chunk_vector
        conv_chunk = flat_conv_chunk_vector

        # TODO: implement vector rev rebin and switch to it
        # rev_rebin_chunk = flat_rev_rebin_chunk_vector

        rev_rebin_chunk = flat_rev_rebin_chunk_single
    elif conf['detector_shape'] == 'curved':

        # use vectorized forms

        diff_chunk = curved_diff_chunk_vector
        fwd_rebin_chunk = curved_fwd_rebin_chunk_vector
        conv_chunk = curved_conv_chunk_vector

        # TODO: implement vector rev rebin and switch to it
        # rev_rebin_chunk = curved_rev_rebin_chunk_vector

        rev_rebin_chunk = curved_rev_rebin_chunk_single

    logging.debug('filtering chunk %d with projections %d to %d' % \
                  (chunk_index, first_proj, last_proj))
    before_chunk = time.time()
    logging.debug('differentiating %d to %d of %d projections' % (first_proj,
                 last_proj, len(input_array)))
    diff_time = diff_chunk(first_proj, last_proj, input_array, diff_array,
                           conf)
    logging.debug('finished diff kernel in %ss' % diff_time)

    # No more use for the extra projection from this point on

    (out_first, out_last) = (first_proj, last_proj)
    fwd_rebin_time = fwd_rebin_chunk(out_first, out_last, diff_array,
            rebin_array, conf)
    logging.debug('finished fwd rebin kernel in %ss' % fwd_rebin_time)
    conv_time = conv_chunk(out_first, out_last, rebin_array, hilbert_array,
                           conv_array, conf)
    logging.debug('finished conv kernel in %ss' % conv_time)
    rev_rebin_time = rev_rebin_chunk(out_first, out_last, conv_array,
            output_array, conf)
    logging.debug('finished rev rebin kernel in %ss' % rev_rebin_time)
    after_chunk = time.time()
    chunk_time = after_chunk - before_chunk
    logging.debug('finished filter kernel in %ss' % chunk_time)
    return chunk_time


def backproject_chunk(
    chunk_index,
    first_proj,
    last_proj,
    first_z,
    last_z,
    input_array,
    row_mins_array,
    row_maxs_array,
    output_array,
    conf,
    ):
    """Run backprojection on chunk of projections keeping the results in
    output_array.

    Parameters
    ----------
    chunk_index : int
        Index of chunk in chunked backprojection.
    first_proj : int
        Index of first projection to include in chunked backprojection.
    last_proj : int
        Index of last projection to include in chunked backprojection.
    first_z : int
        Index of first z voxels to include in chunked backprojection.
    last_z : int
        Index of last z voxels to include in chunked backprojection.
    input_array : ndarray
        Input array.
    row_mins_array : ndarray
        Row interpolation helper array.
    row_maxs_array : ndarray
        Row interpolation helper array.
    output_array : ndarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Backprojection time.
    """

    if conf['detector_shape'] == 'flat':
        backproj_chunk = flat_backproject_chunk
    elif conf['detector_shape'] == 'curved':
        backproj_chunk = curved_backproject_chunk
    return backproj_chunk(chunk_index, first_proj, last_proj, first_z, last_z,
                          input_array, row_mins_array, row_maxs_array,
                          output_array, conf)
