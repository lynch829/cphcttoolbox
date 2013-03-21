#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - cuda specific katsevich reconstruction kernels
# Copyright (C) 2011-2012  The CT-Toolbox Project lead by Brian Vinter
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

from cphct.log import logging

# These are basic numpy functions exposed through npy to use same numpy

from cphct.npycore import int32, sqrt


def filter_chunk(
    first,
    last,
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
    first : int
        Index of first projection to include in chunked filtering.
    last : int
        Index of last projection to include in chunked filtering.
    input_array : gpuarray
        Input array.
    diff_array : gpuarray
        Differentiation helper array.
    rebin_array : gpuarray
        Rebinning helper array.
    conv_array : gpuarray
        Convolution helper array.
    hilbert_array : gpuarray
        Hilbert convolution helper array.
    output_array : gpuarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Filtering time.
    """

    cuda = conf['gpu_module']
    if conf['detector_shape'] == 'flat':
        diff_chunk = conf['compute']['kernels']['flat_diff_chunk']
        fwd_rebin_chunk = conf['compute']['kernels']['flat_fwd_rebin_chunk']
        convolve_chunk = conf['compute']['kernels']['flat_convolve_chunk']
        rev_rebin_chunk = conf['compute']['kernels']['flat_rev_rebin_chunk']
    elif conf['detector_shape'] == 'curved':
        diff_chunk = conf['compute']['kernels']['curved_diff_chunk']
        fwd_rebin_chunk = conf['compute']['kernels']['curved_fwd_rebin_chunk']
        convolve_chunk = conf['compute']['kernels']['curved_convolve_chunk']
        rev_rebin_chunk = conf['compute']['kernels']['curved_rev_rebin_chunk']
    chunk_projs = last - first

    # We want to divide work so that thread warps access a sequential slice of
    # memory, i.e. pixels in the same column. Thus we use first index inside
    # thread blocks for column index and leave rows and projs to fill the rest.

    # Layouts are calculated for a single projection so we scale grids to fit
    gpu_layouts = conf['app_state']['gpu']['layouts']
    rebin_block, _ = filter_block, _ = gpu_layouts['proj_filter']
    filter_grid = (chunk_projs, conf['detector_rows'] * conf['detector_columns']
                   / (filter_block[0] * filter_block[1]))
    rebin_grid = (chunk_projs, conf['detector_rebin_rows'] * conf['detector_columns']
                   / (rebin_block[0] * rebin_block[1]))
    logging.debug('filter kernels with new layout %s %s'
                 % (filter_block, filter_grid))
    logging.debug('rebin kernels with new layout %s %s' % (rebin_block,
                 rebin_grid))
    
    # We keep data on the gpu for efficiency

    chunk_time = 0.0
    diff_time = diff_chunk(
        int32(first),
        int32(last),
        input_array,
        diff_array,
        block=filter_block,
        grid=filter_grid,
        time_kernel=True,
        )
    cuda.Context.synchronize()
    chunk_time += diff_time
    fwd_rebin_time = fwd_rebin_chunk(
        int32(first),
        int32(last),
        diff_array,
        rebin_array,
        block=rebin_block,
        grid=rebin_grid,
        time_kernel=True,
        )
    cuda.Context.synchronize()
    chunk_time += fwd_rebin_time
    convolve_time = convolve_chunk(
        int32(first),
        int32(last),
        rebin_array,
        hilbert_array,
        conv_array,
        block=rebin_block,
        grid=rebin_grid,
        time_kernel=True,
        )
    cuda.Context.synchronize()
    chunk_time += convolve_time
    rev_rebin_time = rev_rebin_chunk(
        int32(first),
        int32(last),
        conv_array,
        output_array,
        block=filter_block,
        grid=filter_grid,
        time_kernel=True,
        )
    cuda.Context.synchronize()
    chunk_time += rev_rebin_time
    logging.debug('kernel times %s %s %s %s s' % (diff_time,
                  fwd_rebin_time, convolve_time, rev_rebin_time))
    logging.debug('finished filter kernels in %ss' % chunk_time)
    return chunk_time


def backproject_chunk(
    first,
    last,
    input_array,
    row_mins_array,
    row_maxs_array,
    output_array,
    conf,
    ):
    """Run backprojection on chunk of projections keeping the results on the
    gpu.

    Parameters
    ----------
    first : int
        Index of first projection to include in chunked backprojection.
    last : int
        Index of last projection to include in chunked backprojection.
    input_array : gpuarray
        Input array.
    row_mins_array : gpuarray
        Row interpolation helper array.
    row_maxs_array : gpuarray
        Row interpolation helper array.
    output_array : gpuarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : float
        Backprojection time.
    """

    cuda = conf['gpu_module']
    if conf['detector_shape'] == 'flat':
        backproject_chunk = conf['compute']['kernels']['flat_backproj_chunk']
    elif conf['detector_shape'] == 'curved':
        backproject_chunk = conf['compute']['kernels']['curved_backproj_chunk']

    # TODO: replace this custom gpu layout with one generated by helper function
    # ... GPUs don't support block z-dim > 64 so this on limits chunk-size to 64
    if conf['chunk_size'] < conf['gpu_target_threads']:
        xy_product = conf['gpu_target_threads'] / conf['chunk_size']

        # Try splitting evenly in x and y but fall back to (1, xy_product)

        x_parts = y_parts = int(sqrt(xy_product))
        if xy_product * conf['chunk_size'] != conf['gpu_target_threads'
                ] or x_parts * y_parts != xy_product:

            # Not easy to split nicely - reasonable guesstimate

            (x_parts, y_parts) = (1, xy_product)

            # balanced x and y gives best performance

            while y_parts % 2 == 0 and y_parts > x_parts:
                y_parts /= 2
                x_parts *= 2
    else:

        # Enough threads per block already

        (x_parts, y_parts) = (1, 1)
    backproject_block = (x_parts, y_parts, conf['chunk_size'])
    backproject_grid = (conf['x_voxels'] / x_parts, conf['y_voxels']
                        / y_parts)
    logging.debug('backproject kernel with layout %s %s'
                 % (backproject_block, backproject_grid))
    
    chunk_time = backproject_chunk(
        int32(first),
        int32(last),
        input_array,
        row_mins_array,
        row_maxs_array,
        output_array,
        block=backproject_block,
        grid=backproject_grid,
        time_kernel=True,
        )
    cuda.Context.synchronize()
    logging.debug('finished backproject kernel in %ss' % chunk_time)
    return chunk_time
