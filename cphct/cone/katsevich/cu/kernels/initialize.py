#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# initialize - cuda specific initialization helpers
# Copyright (C) 2011-2012  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Numpy specific kernel initialization helper functions"""

from cphct.log import logging
from cphct.npycore import pi, zeros, arange, sin, cos, tan, \
    arctan
from cphct.npycore.io import npy_alloc, get_npy_data, get_npy_size
from cphct.cu.io import cu_alloc, get_cu_data
from cphct.cu.core import get_gpu_layout
from cphct.npycore.misc import linear_coordinates

# Runtime constant variables for use in kernels - keep order in sync with gpu
# init code unless using gpu_kernel_auto_init with the same list.

rt_const = {}

rt_const['int'] = [
    'x_voxels',
    'y_voxels',
    'z_voxels',
    'detector_rows',
    'detector_columns',
    'chunk_size',
    'chunk_projs_offset',
    'kernel_radius',
    'kernel_width',
    ]
    
rt_const['float'] = [
    's_min',
    'delta_s',
    'x_min',
    'delta_x',
    'y_min',
    'delta_y',
    'z_min',
    'delta_z',
    'scan_radius',
    'scan_diameter',
    'fov_radius',
    'half_fan_angle',
    'detector_pixel_height',
    'detector_pixel_span',
    'detector_row_offset',
    'detector_column_offset',
    'detector_rebin_rows',
    'detector_rebin_rows_height',
    'progress_per_radian',
    'pi',
    ]

rt_const['str'] = []


def __get_gpu_layouts(
    conf,
    proj_height,
    proj_width,
    proj_rebin,
    x_voxels,
    y_voxels,
    chunk_size,
    max_gpu_threads_pr_block,
    ):
    """
    Generates GPU layouts for the different steps in the Katsevich algorithm, 
    based on the data layout in each step
    
    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    proj_height : int
       Number of rows in the projection matrix
    proj_width : int
       Number of columns in the projection matrix
    proj_rebin : int
       Number of rebin rows in the projection filtering
    x_voxels : int
       Field of View resolution in x
    y_voxels : int
       Field of View resolution in y
    chunk_size : int
       Field of View chunk size, i.e. number of z-slices per chunk
    max_gpu_threads_pr_block : int
       The maximum number of threads in each GPU block
      
    Returns
    -------
    output : tuple
       Tuple of GPU layouts 
    
    Raises
    ------
    ValueError:
       If unable to generate valid GPU layouts
    """

    # Create projection gpu_layout:
    # iterate over columns for memory coalescing

    gpu_rebin_layout = get_gpu_layout(proj_height, proj_width,
                                      max_gpu_threads_pr_block)

    logging.debug('gpu_rebin_layout: %s' % str(gpu_rebin_layout))

    # Create projection filter gpu_layout:
    # iterate over columns for memory coalescing

    gpu_proj_layout = get_gpu_layout(proj_height, proj_width,
                                     max_gpu_threads_pr_block)

    logging.debug('gpu_proj_layout: %s' % str(gpu_proj_layout))

    # Create backprojection Layout:
    # iterate over z for memory coalescing

    gpu_backproject_layout = get_gpu_layout(y_voxels, chunk_size,
            max_gpu_threads_pr_block)

    logging.debug('gpu_backproject_layout: %s' % str(gpu_backproject_layout))
    

    # We want to divide work so that thread warps access a sequential slice of
    # memory, i.e. pixels/voxels in the same column. Thus for filtering we use
    # first index inside thread blocks for column index and leave rows and
    # projs to fill the rest. Similarly we iterate fastest over z then y and x
    # to improve performance.

    # Layouts are calculated for a single projection so we scale grids to fit

    rebin_block, rebin_grid = gpu_rebin_layout
    proj_block, proj_grid = gpu_proj_layout
    backproject_block, backproject_grid = gpu_backproject_layout

    # GPUs with compute capability 2.0+ and CUDA 4.0+ support 3D grids.
    # Fall back to more compute intensive packed index calculation.
    # GPU typically has x-dim >= y-dim >= z-dim support so fit to order
    #       ... we expect detector_rows >= detector_columns >= chunk_projs
    #       ... and x_voxels >= y_voxels >= chunk_size
    
    gpu_specs = conf['gpu']['specs']
    if gpu_specs['MAX_GRID_DIM_Z'] >= 1:
        logging.debug('using 3D gpu grid')
        proj_grid = (conf['detector_columns'] / proj_block[0],
                     conf['detector_rows'] / proj_block[1],
                     conf['filter_out_projs'])
        rebin_grid = (conf['detector_columns'] / rebin_block[0],
                      conf['detector_rebin_rows'] / rebin_block[1],
                      conf['filter_out_projs'])
        backproject_grid = (conf['x_voxels'],
                            conf['y_voxels'] / backproject_block[1],
                            conf['chunk_size'] / backproject_block[0])
    else:
        proj_grid = (conf['detector_columns'] * conf['detector_rows']
                       / (proj_block[0] * proj_block[1]), conf['chunk_projs'])
        rebin_grid = (conf['detector_columns'] * conf['detector_rebin_rows']
                      / (rebin_block[0] * rebin_block[1]), conf['chunk_projs'])
        backproject_grid = (conf['x_voxels'],
                            conf['chunk_size'] * conf['y_voxels']
                            / (backproject_block[0] * backproject_block[1]))
    logging.debug('filter kernels with layout %s %s' % (proj_block,
                                                        proj_grid))
    logging.debug('rebin kernels with layout %s %s' % (rebin_block,
                                                       rebin_grid))
    logging.debug('backproject kernels with layout %s %s' % (backproject_block,
                                                             backproject_grid))
    gpu_rebin_layout = (rebin_block, rebin_grid)
    gpu_proj_layout = (proj_block, proj_grid)
    gpu_backproject_layout = (backproject_block, backproject_grid)

    return (gpu_rebin_layout, gpu_proj_layout,
            gpu_backproject_layout)

def init_recon(conf, fdt):
    """
    Initialize data structures for cuda Katsevich reconstruction

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    fdt : dtype
        Float data type (internal precision).

    Returns
    -------
    output : dict
       Returns configuration dictionary filled with cuda Katsevich
       data structures
    """

    # Create app_state gpu entry

    conf['app_state']['gpu'] = {}

    # Get gpu module handle

    gpu_module = conf['gpu']['module']
    
    conf['app_state']['gpu']['layouts'] = {}

    (conf['app_state']['gpu']['layouts']['rebin'], conf['app_state'
     ]['gpu']['layouts']['proj'], conf['app_state']['gpu'
     ]['layouts']['backproject']) = __get_gpu_layouts(
        conf,
        conf['detector_rows'],
        conf['detector_columns'],
        conf['detector_rebin_rows'],
        conf['x_voxels'],
        conf['y_voxels'],
        conf['chunk_size'],
        conf['gpu_target_threads'],
        )

    # Create z voxel coordinate array

    npy_alloc(conf, 'z_voxels_coordinates',
              linear_coordinates(conf['z_min'], conf['z_max'],
              conf['z_voxels'], True, fdt))

    # Allocate memory for one reconstruction chunk

    npy_alloc(conf, 'recon_chunk', zeros((conf['chunk_size'],
              conf['y_voxels'], conf['x_voxels']), dtype=fdt))


    # Allocate memory for projection data

    npy_alloc(conf, 'projs_data', zeros((1, conf['detector_rows'],
              conf['detector_columns']), dtype=fdt))



    npy_alloc(conf, 'source_pos', conf['s_min'] + conf['delta_s']
              * (arange(conf['total_projs'], dtype=fdt) + 0.5))

    # Detector pixel center coordinates in both directions, using coordinate
    # system centered in the middle of the detector
    # We include one extra pixel coordinate in ext version for use in
    # interpolation later
    # Please note that the row coordinate formula in (82) of the Noo paper
    # is wrong. It should be Nrows instead of Ncols and fig 4 indicates that
    # the center row coordinates are used which means it should include the
    # '-1' too to get the half pixel offset like for the curved detector.

    npy_alloc(conf, 'row_coords_ext', conf['detector_pixel_height']
              * (arange(conf['detector_rows'] + 1, dtype=fdt)
              + conf['detector_row_offset'] - 0.5
              * (conf['detector_rows'] - 1)))
    row_coords_ext = get_npy_data(conf, 'row_coords_ext')
    row_coords = row_coords_ext[:-1]

    npy_alloc(conf, 'col_coords_ext', conf['detector_pixel_span']
              * (arange(conf['detector_columns'] + 1, dtype=fdt)
              + conf['detector_column_offset'] - 0.5
              * (conf['detector_columns'] - 1)))
    col_coords_ext = get_npy_data(conf, 'col_coords_ext')
    col_coords = col_coords_ext[:-1]

    # TODO: do we need to offset rebin rows like rows?

    npy_alloc(conf, 'rebin_coords', -pi / 2 - conf['half_fan_angle']
              + conf['detector_rebin_rows_height']
              * arange(conf['detector_rebin_rows'], dtype=fdt))
    rebin_coords = get_npy_data(conf, 'rebin_coords')

    # rebinning coordinates

    npy_alloc(conf, 'fwd_rebin_row', zeros((conf['detector_rebin_rows'
              ], conf['detector_columns']), dtype=fdt))
    fwd_rebin_row = get_npy_data(conf, 'fwd_rebin_row')

    # Simplified version of original scale expression:
    # (scan_diameter * progress_per_turn) / (2 * pi * scan_radius)

    rebin_scale = 2 * conf['progress_per_radian']

    # skip last column from differentiation

    for col in xrange(conf['detector_columns']):
        if conf['detector_shape'] == 'flat':
            row = rebin_scale * (rebin_coords + rebin_coords
                                 / tan(rebin_coords) * (col_coords[col]
                                 / conf['scan_diameter']))
        elif conf['detector_shape'] == 'curved':
            row = rebin_scale * (rebin_coords * cos(col_coords[col])
                                 + rebin_coords / tan(rebin_coords)
                                 * sin(col_coords[col]))
        fwd_rebin_row[:, col] = row

    # Hilbert helper values

    npy_alloc(conf, 'hilbert_ideal', zeros(conf['kernel_width'],
              dtype=fdt))
    hilbert_ideal = get_npy_data(conf, 'hilbert_ideal')

    # We use a simplified hilbert kernel for now

    kernel_radius = conf['kernel_radius']
    for i in xrange(conf['kernel_width']):
        hilbert_ideal[i] = (1.0 - cos(pi * (i - kernel_radius - 0.5))) \
            / (pi * (i - kernel_radius - 0.5))

    # Tam-Danielsson boundaries in projections
    # We use extended col coords to allow full interpolation in back projection

    if conf['detector_shape'] == 'flat':
        min_help = 2 * arctan(conf['scan_diameter'] / col_coords_ext)
        max_help = min_help.copy()
        min_help[conf['detector_columns'] / 2:] -= 2 * pi
        max_help[:conf['detector_columns'] / 2] += 2 * pi
        proj_row_mins = conf['progress_per_turn'] / pi * min_help / \
                        (1 - cos(min_help))
        proj_row_maxs = conf['progress_per_turn'] / pi * max_help / \
                        (1 - cos(max_help))
    elif conf['detector_shape'] == 'curved':
        proj_row_mins = -conf['progress_per_turn'] / pi * \
                        (pi / 2 + col_coords_ext) / cos(col_coords_ext)
        proj_row_maxs = conf['progress_per_turn'] / pi * \
                        (pi / 2 - col_coords_ext) / cos(col_coords_ext)
    npy_alloc(conf, 'proj_row_mins', proj_row_mins)
    npy_alloc(conf, 'proj_row_maxs', proj_row_maxs)

    # Matrices for storage in host memory - init with zeros to make sure they
    # are fully initialized for mem size checks

    npy_alloc(conf, 'filter_in', zeros((conf['filter_in_projs'],
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))

    # TODO: We don't strictly need dedicated arrays for each filter step
    # ...we could use just two of filter_out size and alternate between them
    # that would save us quite some gpu mem but make checksum a bit cumbersome

    npy_alloc(conf, 'filter_diff', zeros((conf['filter_out_projs'],
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))
    npy_alloc(conf, 'filter_rebin', zeros((conf['filter_out_projs'],
              conf['detector_rebin_rows'], conf['detector_columns']),
              dtype=fdt))
    npy_alloc(conf, 'filter_conv', zeros((conf['filter_out_projs'],
              conf['detector_rebin_rows'], conf['detector_columns']),
              dtype=fdt))
    npy_alloc(conf, 'filter_out', zeros((conf['filter_out_projs'],
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))

    # Buffer filtered results for use in back projection
    # Make sure input buffer can hold chunk projs and one rotation on each side
    # We use input_projs for storing the projections for a backprojection chunk
    # in host memory and input_chunk for the projections used by a single
    # backprojection step.

    input_buffer_projs = (2 + conf['chunk_projs']
                          / conf['projs_per_turn']) \
        * conf['projs_per_turn']
    npy_alloc(conf, 'input_buffer', zeros((input_buffer_projs,
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))
    npy_alloc(conf, 'input_projs', zeros((conf['chunk_projs'],
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))
    npy_alloc(conf, 'input_chunk', zeros((conf['backproject_in_projs'],
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))
    npy_alloc(conf, 'output_chunk', zeros((conf['x_voxels'],
              conf['y_voxels'], conf['chunk_size']), dtype=fdt))

    if conf['checksum']:

        # We never need more than input_chunk size in checks

        input_chunk = get_npy_data(conf, 'input_chunk')
        npy_alloc(conf, 'check_dest', zeros(input_chunk.size,
                  input_chunk.dtype))

    # Matrices for storage in device memory

    total_size = 0
    for buf_name in ('filter_in', 'filter_diff', 'filter_rebin',
                     'filter_conv', 'filter_out', 'input_chunk',
                     'output_chunk', 'hilbert_ideal', 'proj_row_mins',
                     'proj_row_maxs'):
        buf_size = get_npy_size(conf, buf_name)
        logging.debug('cu allocing %dB for %s' % (buf_size, buf_name))
        cu_alloc(conf, buf_name, gpu_module.mem_alloc(buf_size), buf_size)
        total_size += buf_size
        logging.debug('cu alloc size for %s is %s - total %s' % \
                      (buf_name, buf_size, total_size))

    # Get handles to GPU structures and copy conf/helpers to constant memory

    conf['compute'] = {'kernels': {}}
    compute_kernels = conf['compute']['kernels']
    gpu_hilbert_ideal = get_cu_data(conf, 'hilbert_ideal')
    gpu_proj_row_mins = get_cu_data(conf, 'proj_row_mins')
    gpu_proj_row_maxs = get_cu_data(conf, 'proj_row_maxs')

    gpu_module.memcpy_htod(gpu_hilbert_ideal, hilbert_ideal)
    gpu_module.memcpy_htod(gpu_proj_row_mins, proj_row_mins)
    gpu_module.memcpy_htod(gpu_proj_row_maxs, proj_row_maxs)

    cu_kernels = conf['cu_kernels']
    compute_kernels['flat_diff_chunk'] = cu_kernels.get_function(
        'flat_diff_chunk')
    compute_kernels['flat_fwd_rebin_chunk'] = cu_kernels.get_function(
        'flat_fwd_rebin_chunk')
    compute_kernels['flat_convolve_chunk'] = cu_kernels.get_function(
        'flat_convolve_chunk')
    compute_kernels['flat_rev_rebin_chunk'] = cu_kernels.get_function(
        'flat_rev_rebin_chunk')
    compute_kernels['flat_backproj_chunk'] = cu_kernels.get_function(
        'flat_backproject_chunk')
    compute_kernels['curved_diff_chunk'] = cu_kernels.get_function(
        'curved_diff_chunk')
    compute_kernels['curved_fwd_rebin_chunk'] = cu_kernels.get_function(
        'curved_fwd_rebin_chunk')
    compute_kernels['curved_convolve_chunk'] = cu_kernels.get_function(
        'curved_convolve_chunk')
    compute_kernels['curved_rev_rebin_chunk'] = cu_kernels.get_function(
        'curved_rev_rebin_chunk')
    compute_kernels['curved_backproj_chunk'] = cu_kernels.get_function(
        'curved_backproject_chunk')
    compute_kernels['checksum'] = cu_kernels.get_function('checksum_array')

    return conf
