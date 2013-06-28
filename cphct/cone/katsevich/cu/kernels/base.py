#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - cuda specific katsevich reconstruction kernels
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

from cphct.log import logging

# These are basic numpy functions exposed through npycore to use same numpy

from cphct.npycore import int32, sqrt


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
    in output_array. The first_proj and last_proj argument are projection
    indices for the first and last input projections. The differentiation step
    uses an extra projection, so filtering produces filtered projections with
    indices from first_proj to last_proj - 1.

    Parameters
    ----------
    chunk_index : int
        Index of chunk in chunked backprojection.
    first_proj : int
        Index of first projection to include in chunked filtering.
    last_proj : int
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

    cuda = conf['gpu']['module']
    gpu_layouts = conf['app_state']['gpu']['layouts']
    rebin_block, rebin_grid = gpu_layouts['rebin']
    proj_block, proj_grid = gpu_layouts['proj']
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

    # We keep data on the gpu for efficiency

    chunk_time = 0.0
    diff_time = diff_chunk(
        int32(first_proj),
        int32(last_proj),
        input_array,
        diff_array,
        block=proj_block,
        grid=proj_grid,
        time_kernel=True,
        )
    cuda.Context.synchronize()
    chunk_time += diff_time
    fwd_rebin_time = fwd_rebin_chunk(
        int32(first_proj),
        int32(last_proj),
        diff_array,
        rebin_array,
        block=rebin_block,
        grid=rebin_grid,
        time_kernel=True,
        )
    cuda.Context.synchronize()
    chunk_time += fwd_rebin_time
    convolve_time = convolve_chunk(
        int32(first_proj),
        int32(last_proj),
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
        int32(first_proj),
        int32(last_proj),
        conv_array,
        output_array,
        block=proj_block,
        grid=proj_grid,
        time_kernel=True,
        )
    cuda.Context.synchronize()
    chunk_time += rev_rebin_time
    logging.debug('kernel times %s %s %s %s s' % (diff_time,
                  fwd_rebin_time, convolve_time, rev_rebin_time))
    logging.debug('finished filter kernels in %ss' % chunk_time)
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
    """Run backprojection on chunk of projections keeping the results on the
    gpu.

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

    cuda = conf['gpu']['module']
    gpu_layouts = conf['app_state']['gpu']['layouts']
    backproject_block, backproject_grid = gpu_layouts['backproject']
    if conf['detector_shape'] == 'flat':
        backproject_chunk = conf['compute']['kernels']['flat_backproj_chunk']
    elif conf['detector_shape'] == 'curved':
        backproject_chunk = conf['compute']['kernels']['curved_backproj_chunk']

    chunk_time = backproject_chunk(
        int32(chunk_index),
        int32(first_proj),
        int32(last_proj),
        int32(first_z),
        int32(last_z),
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
