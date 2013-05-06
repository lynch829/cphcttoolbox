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
    # GPUs with compute capability 2.0+ and CUDA 4.0+ support 3D grids.
    # Fall back to more compute intensive packed index calculation.
    # GPU typically has x-dim >= y-dim >= z-dim support so fit to order
    #       ... we expect detector_rows >= detector_columns >= chunk_projs
    gpu_specs = conf['app_state']['gpu']['specs']
    if gpu_specs['MAX_GRID_DIM_Z'] >= 1:
        logging.debug('using 3D gpu grid')
        filter_grid = (conf['detector_columns'] / filter_block[0],
                       conf['detector_rows'] / filter_block[1],
                       chunk_projs)
        rebin_grid = (conf['detector_columns'] / rebin_block[0],
                      conf['detector_rebin_rows'] / rebin_block[1],
                      chunk_projs)
    else:
        filter_grid = (conf['detector_columns'] * conf['detector_rows']
                       / (filter_block[0] * filter_block[1]), chunk_projs)
        rebin_grid = (conf['detector_columns'] * conf['detector_rebin_rows']
                      / (rebin_block[0] * rebin_block[1]), chunk_projs)
    logging.debug('filter kernels with layout %s %s'
                 % (filter_block, filter_grid))
    logging.debug('rebin kernels with layout %s %s' % (rebin_block,
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

    # We want to divide work so that thread warps access a sequential slice of
    # memory, i.e. voxels in adjacent z positions. However it turns out that
    # iterating fastest over y and then z significantly improves performance.

    # Layouts are calculated for a single slice so we scale grids to fit
    gpu_layouts = conf['app_state']['gpu']['layouts']
    backproject_block, _ = gpu_layouts['backproject']
    # GPUs with compute capability 2.0+ and CUDA 4.0+ support 3D grids.
    # Fall back to slightly more compute intensive packed index calculation.
    # GPU typically has x-dim >= y-dim >= z-dim support so fit to order
    #       ... we expect x_voxels >= y_voxels >= chunk_size
    gpu_specs = conf['app_state']['gpu']['specs']
    if gpu_specs['MAX_GRID_DIM_Z'] >= 1:
        backproject_grid = (conf['x_voxels'],
                            conf['y_voxels'] / backproject_block[1],
                            conf['chunk_size'] / backproject_block[0])
    else:
        backproject_grid = (conf['x_voxels'],
                            conf['chunk_size'] * conf['y_voxels']
                            / (backproject_block[0] * backproject_block[1]))
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
