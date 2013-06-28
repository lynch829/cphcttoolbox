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

from cphct.log import logging
from cphct.npycore import allowed_data_types
from cphct.npycore.io import npy_alloc, get_npy_data
from cphct.npycore.misc import linear_coordinates
from cphct.cu.io import cu_alloc
from cphct.cu.core import gpuarray, get_gpu_layout
from cphct.cone.fdk.npycore.kernels import generate_combined_matrix, \
    generate_detector_boundingboxes
import pyfft.cuda

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
    'proj_filter_width',
    ]

rt_const['float'] = [
    'source_distance',
    'detector_distance',
    'detector_pixel_height',
    'detector_pixel_width',
    'detector_column_shift',
    'detector_row_shift',
    'volume_weight_factor',
    ]

rt_const['str'] = ['detector_shape', 'volume_weight']


def __get_gpu_layouts(
    proj_height,
    proj_width,
    proj_filter_width,
    x_voxels,
    y_voxels,
    max_gpu_threads_pr_block,
    ):
    """
    Generates GPU layouts for the different steps in the FDK algorithm, 
    based on the data layout in each step
    
    Parameters
    ----------
    proj_height : int
       Number of rows in the projection matrix
    proj_width : int
       Number of columns in the projection matrix
    proj_filter_width : int
       Number of elements in the projection filter array
    x_voxels : int
       Field of View resolution in x
    y_voxels : int
       Field of View resolution in y
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

    # Create projection gpu_layout

    gpu_proj_layout = get_gpu_layout(proj_height, proj_width,
            max_gpu_threads_pr_block)

    logging.debug('gpu_proj_layout: %s' % str(gpu_proj_layout))

    # Create projection filter gpu_layout
    # The projection filter layout is complex
    # and therefore have a complex and a real part

    gpu_proj_filter_layout = get_gpu_layout(proj_height,
            proj_filter_width * 2, max_gpu_threads_pr_block)

    logging.debug('gpu_proj_filter_layout: %s'
                  % str(gpu_proj_filter_layout))

    # Create backprojection Layout

    gpu_backproject_layout = get_gpu_layout(y_voxels, x_voxels,
            max_gpu_threads_pr_block)

    logging.debug('gpu_backproject_layout: %s'
                  % str(gpu_backproject_layout))

    return (gpu_proj_layout, gpu_proj_filter_layout,
            gpu_backproject_layout)


def init_recon(conf, fdt):
    """
    Initialize data structures for cuda FDK reconstruction

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    fdt : dtype
        Float data type (internal precision).

    Returns
    -------
    output : dict
       Returns configuration dictionary filled with 
       numpy and cuda FDK reconstruction data structures
    """

    # Create app_state gpu entry

    conf['app_state']['gpu'] = {}

    # Get complex data type

    cdt = allowed_data_types[conf['complex_precision']]

    # Get gpu module handle

    gpu_module = conf['gpu']['module']

    # Get detector boundingboxes for each recon chunk

    detector_boundingboxes = generate_detector_boundingboxes(conf, fdt)
    npy_alloc(conf, 'detector_boundingboxes', detector_boundingboxes)

    # Find the maximal number of detector rows covered by a single recon chunk

    max_proj_rows = int((detector_boundingboxes[:, 0, 1]
                        - detector_boundingboxes[:, 0, 0]).max())
    conf['app_state']['projs']['max_rows'] = max_proj_rows

    # Generate GPU layouts for each processing task

    conf['app_state']['gpu']['layouts'] = {}

    (conf['app_state']['gpu']['layouts']['proj'], conf['app_state'
     ]['gpu']['layouts']['proj_filter'], conf['app_state']['gpu'
     ]['layouts']['backproject']) = __get_gpu_layouts(
        max_proj_rows,
        conf['detector_columns'],
        conf['proj_filter_width'],
        conf['x_voxels'],
        conf['y_voxels'],
        conf['gpu_target_threads'],
        )

    # Allocate memory for detector_offset

    gpu_proj_row_offset = gpuarray.zeros(1,
            dtype=allowed_data_types['uint32'])
    cu_alloc(conf, 'proj_row_offset', gpu_proj_row_offset,
             gpu_proj_row_offset.nbytes)

    # Allocate memory for the transform_matrix on GPU

    gpu_transform_matrix = gpuarray.zeros((3, 4), dtype=fdt)
    cu_alloc(conf, 'transform_matrix', gpu_transform_matrix,
             gpu_transform_matrix.nbytes)

    # Create gpu_proj_weight_matrix

    if conf['proj_weight'] != 'skip':
        gpu_proj_weight_matrix = gpuarray.to_gpu(get_npy_data(conf,
                'proj_weight_matrix'))
        cu_alloc(conf, 'proj_weight_matrix', gpu_proj_weight_matrix,
                 gpu_proj_weight_matrix.nbytes)

    # Create gpu_proj_filter_array

    if conf['proj_filter'] != 'skip':
        gpu_proj_filter_array = gpuarray.to_gpu(get_npy_data(conf,
                'proj_filter_array'))
        cu_alloc(conf, 'proj_filter_array', gpu_proj_filter_array,
                 gpu_proj_filter_array.nbytes)

    # Get combined matrix,
    # only use X,Y combination [:-2] in the GPU version
    # as Z coordinates are handled separately

    combined_matrix = generate_combined_matrix(
        conf['x_min'],
        conf['x_max'],
        conf['x_voxels'],
        conf['y_min'],
        conf['y_max'],
        conf['y_voxels'],
        fdt,
        )[:-2]

    # Move combined matrix to GPU,

    gpu_combined_matrix = gpuarray.to_gpu(combined_matrix)
    cu_alloc(conf, 'combined_matrix', gpu_combined_matrix,
             gpu_combined_matrix.nbytes)

    # Create z voxel coordinate array

    z_voxel_coordinates = linear_coordinates(conf['z_min'], conf['z_max'
            ], conf['z_voxels'], True, fdt)

    gpu_z_voxel_coordinates = gpuarray.to_gpu(z_voxel_coordinates)
    cu_alloc(conf, 'z_voxel_coordinates', gpu_z_voxel_coordinates,
             gpu_z_voxel_coordinates.nbytes)

    # Allocate memory for projection angle

    gpu_proj_angle_rad = gpuarray.zeros(1, dtype=fdt)
    cu_alloc(conf, 'proj_angle_rad', gpu_proj_angle_rad,
             gpu_proj_angle_rad.nbytes)

    # Allocate pinned memory for projection data, please note that this is a
    # *host* memory allocation even though we use gpu_module call.

    projs_data = gpu_module.pagelocked_zeros((conf['proj_chunk_size'],
            conf['detector_rows'], conf['detector_columns']), dtype=fdt)

    npy_alloc(conf, 'projs_data', projs_data)

    gpu_projs_data = gpuarray.zeros((conf['proj_chunk_size'],
                                    max_proj_rows,
                                    conf['detector_columns']),
                                    dtype=fdt)

    cu_alloc(conf, 'projs_data', gpu_projs_data, gpu_projs_data.nbytes)

    # Allocate memory for fft projection on GPU,
    # the frequency domain representation has an imaginary part,
    # therefore the memory consumption is doubled
    # in order to contain both a real and imaginary part each float

    gpu_complex_proj = gpuarray.zeros((max_proj_rows,
            conf['proj_filter_width']), dtype=cdt)
    cu_alloc(conf, 'complex_proj', gpu_complex_proj,
             gpu_complex_proj.nbytes)

    # Create fft plan used for prefiltering

    fft_plan = pyfft.cuda.Plan(conf['proj_filter_width'], dtype=cdt,
                               fast_math=True)

    conf['app_state']['gpu']['fft_plan'] = fft_plan

    # Allocate memory for chunk index

    gpu_chunk_index = gpuarray.zeros(1,
            dtype=allowed_data_types['uint32'])
    cu_alloc(conf, 'chunk_index', gpu_chunk_index,
             gpu_chunk_index.nbytes)

    # Allocate pinned memory for one reconstruction chunk
    # http://documen.tician.de/pycuda/driver.html#pagelocked-host-memory

    recon_chunk = gpu_module.pagelocked_zeros((conf['chunk_size'],
            conf['y_voxels'], conf['x_voxels']), dtype=fdt)
    npy_alloc(conf, 'recon_chunk', recon_chunk)

    gpu_recon_chunk = gpuarray.zeros(recon_chunk.shape, dtype=fdt)
    cu_alloc(conf, 'recon_chunk', gpu_recon_chunk,
             gpu_recon_chunk.nbytes)

    # Allocate memory for volume weighting

    gpu_volume_weight_matrix = gpuarray.zeros((conf['y_voxels'],
            conf['x_voxels']), dtype=fdt)
    cu_alloc(conf, 'volume_weight_matrix', gpu_volume_weight_matrix,
             gpu_volume_weight_matrix.nbytes)

    return conf


