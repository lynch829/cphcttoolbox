#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - CUDA specific katsevich reconstruction kernels
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

"""Spiral cone beam CT CUDA kernels using the Katsevich algorithm"""

from cphct.log import logging
from cphct.misc import timelog

from cphct.cu.core import gpu_alloc_from_array
from cphct.npycore import allowed_data_types


def filter_chunk(
    first_proj,
    last_proj,
    input_array,
    diff_array,
    rebin_array,
    proj_filter_array,
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
    proj_filter_array : gpuarray
        Hilbert convolution helper array.
    output_array : gpuarray
        Output array.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : gpuarray
        Filtered chunk of projections.
    """

    int32 = allowed_data_types['int32']

    active_gpu_id = conf['gpu']['active_id']
    gpu_stream = conf['app_state']['gpu']['streams'][active_gpu_id]

    gpu_layouts = conf['app_state']['gpu']['layouts']
    (rebin_block, rebin_grid) = gpu_layouts['rebin']
    (proj_block, proj_grid) = gpu_layouts['proj']
    prepared_kernels = conf['app_state']['gpu']['prepared_kernels']

    if conf['detector_shape'] == 'flat':
        diff_chunk = prepared_kernels['flat_diff_chunk']
        fwd_rebin_chunk = prepared_kernels['flat_fwd_rebin_chunk']
        convolve_chunk = prepared_kernels['flat_convolve_chunk']
        rev_rebin_chunk = prepared_kernels['flat_rev_rebin_chunk']
    elif conf['detector_shape'] == 'curved':
        diff_chunk = prepared_kernels['curved_diff_chunk']
        fwd_rebin_chunk = prepared_kernels['curved_fwd_rebin_chunk']
        convolve_chunk = prepared_kernels['curved_convolve_chunk']
        rev_rebin_chunk = prepared_kernels['curved_rev_rebin_chunk']

    # We keep data on the gpu for efficiency

    if conf['log_level'] == logging.DEBUG:
        timelog.set(conf, 'default', 'diff_chunk', barrier=True)

    diff_chunk.prepared_async_call(
        proj_grid,
        proj_block,
        gpu_stream,
        int32(first_proj),
        int32(last_proj),
        gpu_alloc_from_array(input_array),
        gpu_alloc_from_array(diff_array),
        )

    if conf['log_level'] == logging.DEBUG:
        timelog.log(conf, 'default', 'diff_chunk', barrier=True)
        timelog.set(conf, 'default', 'fwd_rebin_chunk')

    fwd_rebin_chunk.prepared_async_call(
        rebin_grid,
        rebin_block,
        gpu_stream,
        int32(first_proj),
        int32(last_proj),
        gpu_alloc_from_array(diff_array),
        gpu_alloc_from_array(rebin_array),
        )

    if conf['log_level'] == logging.DEBUG:
        timelog.log(conf, 'default', 'fwd_rebin_chunk', barrier=True)
        timelog.set(conf, 'default', 'convolve_chunk')

    convolve_chunk.prepared_async_call(
        rebin_grid,
        rebin_block,
        gpu_stream,
        int32(first_proj),
        int32(last_proj),
        gpu_alloc_from_array(rebin_array),
        gpu_alloc_from_array(proj_filter_array),
        gpu_alloc_from_array(conv_array),
        )

    if conf['log_level'] == logging.DEBUG:
        timelog.log(conf, 'default', 'convolve_chunk', barrier=True)
        timelog.set(conf, 'default', 'rev_rebin_chunk')

    rev_rebin_chunk.prepared_async_call(
        proj_grid,
        proj_block,
        gpu_stream,
        int32(first_proj),
        int32(last_proj),
        gpu_alloc_from_array(conv_array),
        gpu_alloc_from_array(output_array),
        )

    if conf['log_level'] == logging.DEBUG:
        timelog.log(conf, 'default', 'rev_rebin_chunk', barrier=True)

        logging.debug('filter kernel times %s %s %s %s s'
                      % (timelog.get(conf, 'default', 'diff_chunk'),
                      timelog.get(conf, 'default', 'fwd_rebin_chunk'),
                      timelog.get(conf, 'default', 'convolve_chunk'),
                      timelog.get(conf, 'default', 'rev_rebin_chunk')))

        logging.debug('finished filter kernels in %ss'
                      % (timelog.get(conf, 'default', 'diff_chunk')
                      + timelog.get(conf, 'default', 'fwd_rebin_chunk')
                      + timelog.get(conf, 'default', 'convolve_chunk')
                      + timelog.get(conf, 'default', 'rev_rebin_chunk'
                      )))

    return output_array


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
    output : gpuarray
        Backprojected volume chunk.
    """

    int32 = allowed_data_types['int32']

    gpu_layouts = conf['app_state']['gpu']['layouts']

    active_gpu_id = conf['gpu']['active_id']
    gpu_stream = conf['app_state']['gpu']['streams'][active_gpu_id]

    (backproject_block, backproject_grid) = gpu_layouts['backproject']
    prepared_kernels = conf['app_state']['gpu']['prepared_kernels']

    if conf['detector_shape'] == 'flat':
        backproject_chunk = prepared_kernels['flat_backproject_chunk']
    elif conf['detector_shape'] == 'curved':
        backproject_chunk = prepared_kernels['curved_backproject_chunk']

    if conf['log_level'] == logging.DEBUG:
        timelog.set(conf, 'default', 'backproject_chunk', barrier=True)

    backproject_chunk.prepared_async_call(
        backproject_grid,
        backproject_block,
        gpu_stream,
        int32(chunk_index),
        int32(first_proj),
        int32(last_proj),
        int32(first_z),
        int32(last_z),
        gpu_alloc_from_array(input_array),
        gpu_alloc_from_array(row_mins_array),
        gpu_alloc_from_array(row_maxs_array),
        gpu_alloc_from_array(output_array),
        )

    if conf['log_level'] == logging.DEBUG:
        timelog.log(conf, 'default', 'backproject_chunk', barrier=True)

        logging.debug('finished backproject kernel in %ss'
                      % timelog.get(conf, 'default', 'backproject_chunk'
                      ))

    return output_array


