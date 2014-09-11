#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - CUDA specific FDK reconstruction kernels
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

"""Step and shoot cone beam CT CUDA kernels using the FDK algorithm"""

from cphct.npycore import real, radians, allowed_data_types

from cphct.npycore.io import get_npy_data, save_auto
from cphct.npycore.utils import log_checksum
from cphct.cu.core import gpu_array_alloc_offset, gpu_alloc_from_array
from cphct.cu.io import get_cu_data
from cphct.cone.fdk.npycore.kernels import generate_transform_matrix
from cphct.log import logging
from cphct.misc import timelog


def weight_proj(
    gpu_proj,
    proj_row_offset,
    gpu_weight,
    gpu_kernel,
    gpu_stream,
    gpu_layouts,
    ):
    """
    Invoke GPU projection data weighting
    
    Parameters
    ----------
    gpu_proj : gpuarray
       Projection data
    proj_row_offset : uint32
       Projection row offset for current chunk
    gpu_weight : gpuarray
       Projection weight
    gpu_kernel : pycuda.driver.Function
       Compiled and prepared CUDA kernel
    gpu_stream : pycda.driver.Stream
       Stream used for async prepared execution
    gpu_layouts : dict
       Grid and block layouts for kernel execution
       
    Returns
    -------
    output : gpuarray
       GPU weighted complex projection
    """

    (block, grid) = gpu_layouts['proj']

    gpu_kernel.prepared_async_call(
        grid,
        block,
        gpu_stream,
        gpu_alloc_from_array(gpu_proj),
        proj_row_offset,
        gpu_alloc_from_array(gpu_weight),
        shared_size=0,
        )

    return gpu_proj


def proj_to_complex(
    gpu_complex_proj,
    gpu_proj,
    nr_proj_rows,
    gpu_kernel,
    gpu_stream,
    gpu_layouts,
    ):
    """
    Transforms projection float data to a complex data
    
    Parameters
    ----------
    gpu_complex_proj : gpuarray
       Complex projection data
    gpu_proj : gpuarray
       Float projection data
    nr_proj_rows : uint32
       Number of active projection rows
    gpu_kernel : pycuda.driver.Function
       Compiled and prepared CUDA kernel
    gpu_stream : pycda.driver.Stream
       Stream used for async prepared execution
    gpu_layouts : dict
       Grid and block layouts for kernel execution
       
    Returns
    -------
    output : gpuarray
       GPU complex projection data
    """

    (block, grid) = gpu_layouts['proj_to_complex']

    gpu_kernel.prepared_async_call(
        grid,
        block,
        gpu_stream,
        gpu_alloc_from_array(gpu_complex_proj),
        gpu_alloc_from_array(gpu_proj),
        nr_proj_rows,
        shared_size=0,
        )

    return gpu_complex_proj


def filter_proj(
    gpu_complex_proj,
    gpu_filter,
    gpu_kernel,
    gpu_stream,
    gpu_layouts,
    ):
    """
    Invoke GPU projection filtering
    
    Parameters
    ----------
    gpu_complex_proj: gpuarray
       Complex projection data
    gpu_filter : gpuarray
       Filter array
    gpu_kernel : pycuda.driver.Function
       Compiled and prepared CUDA kernel
    gpu_stream : pycda.driver.Stream
       Stream used for async prepared execution
    gpu_layouts : dict
       Grid and block layouts for kernel execution
       
    Returns
    -------
    output : gpuarray
       GPU filtered complex projection
    """

    (block, grid) = gpu_layouts['proj_filter']

    gpu_kernel.prepared_async_call(
        grid,
        block,
        gpu_stream,
        gpu_alloc_from_array(gpu_complex_proj),
        gpu_alloc_from_array(gpu_filter),
        shared_size=0,
        )

    return gpu_complex_proj


def complex_to_proj(
    gpu_proj,
    gpu_complex_proj,
    gpu_kernel,
    gpu_stream,
    gpu_layouts,
    ):
    """
    Transforms projection complex data to a float data
    
    Parameters
    ----------
    gpu_proj : gpuarray
       float projection data
    gpu_complex_proj : gpuarray
       complex projection data
    gpu_kernel : pycuda.driver.Function
       Compiled and prepared CUDA kernel
    gpu_stream : pycda.driver.Stream
       Stream used for async prepared execution
    gpu_layouts : dict
       Grid and block layouts for kernel execution
       
    Returns
    -------
    output : gpuarray
       GPU float projection data
    """

    (block, grid) = gpu_layouts['proj']

    gpu_kernel.prepared_async_call(
        grid,
        block,
        gpu_stream,
        gpu_alloc_from_array(gpu_proj),
        gpu_alloc_from_array(gpu_complex_proj),
        shared_size=0,
        )

    return gpu_proj


def generate_volume_weight(
    gpu_volume_weight_matrix,
    gpu_combined_matrix,
    proj_angle_rad,
    gpu_kernel,
    gpu_stream,
    gpu_layouts,
    ):
    """
    Invoke GPU volume weight generator
    
    Parameters
    ----------
    gpu_volume_weight_matrix : gpuarray
       Volume weight matrix
    gpu_combined_matrix : gpuarray
       Combined matrix
    proj_angle_rad : float32
       Projection angle in radians
    gpu_kernel : pycuda.driver.Function
       Compiled and prepared CUDA kernel
    gpu_stream : pycda.driver.Stream
       Stream used for async prepared execution
    gpu_layouts : dict
       Grid and block layouts for kernel execution
       
    Returns
    -------
    output : gpuarray
       GPU volume weight matrix
    """

    (block, grid) = gpu_layouts['backproject']

    gpu_kernel.prepared_async_call(
        grid,
        block,
        gpu_stream,
        gpu_alloc_from_array(gpu_volume_weight_matrix),
        gpu_alloc_from_array(gpu_combined_matrix),
        proj_angle_rad,
        shared_size=0,
        )

    return gpu_volume_weight_matrix


def backproject_proj(
    gpu_recon_chunk,
    gpu_proj,
    proj_row_offset,
    chunk_index,
    gpu_z_voxel_coordinates,
    gpu_transform_matrix,
    gpu_combined_matrix,
    gpu_volume_weight_matrix,
    gpu_kernel,
    gpu_stream,
    gpu_layouts,
    ):
    """
    Invoke GPU backprojection
    
    Parameters
    ----------
    gpu_recon_chunk : gpuarray
       Reconstructed volume chunk
    gpu_proj : gpuarray
       Projection data to reconstruct
    proj_row_offset : uint32
       Projection row offset for current chunk
    chunk_index : uint32
       Chunk index to reconstruct
    gpu_z_voxel_coordinates : gpuarray
       Array with z voxel coordinates
    gpu_combined_matrix : gpuarray
       Matrix with x,y voxel coordinates
    gpu_volume_weight_matrix : gpuarray
       Matrix with volume weights
    gpu_kernel : pycuda.driver.Function
       Compiled and prepared CUDA kernel
    gpu_stream : pycda.driver.Stream
       Stream used for async prepared execution
    gpu_layouts : dict
       Grid and block layouts for kernel execution
       
    Returns
    -------
    output : gpuarray
       Reconstructed volume chunk
    """

    (block, grid) = gpu_layouts['backproject']

    gpu_kernel.prepared_async_call(
        grid,
        block,
        gpu_stream,
        gpu_alloc_from_array(gpu_recon_chunk),
        gpu_alloc_from_array(gpu_proj),
        proj_row_offset,
        chunk_index,
        gpu_alloc_from_array(gpu_z_voxel_coordinates),
        gpu_alloc_from_array(gpu_transform_matrix),
        gpu_alloc_from_array(gpu_combined_matrix),
        gpu_alloc_from_array(gpu_volume_weight_matrix),
        shared_size=0,
        )

    return gpu_recon_chunk


def reconstruct_proj(conf, proj, fdt):
    """Reconstructs a single projection
    conf : dict
        A dictionary of configuration options.
    proj : dict
        Projection dictionary containing meta infomation and data
    fdt : dtype
        Float precision

    Returns
    -------
    output : dict
        The dictionary of configuration options.
    """

    # Get data types

    uint32 = allowed_data_types['uint32']

    # Get gpu streams

    active_gpu_id = conf['gpu']['active_id']
    gpu_stream = conf['app_state']['gpu']['streams'][active_gpu_id]

    # Get gpu layouts

    gpu_layouts = conf['app_state']['gpu']['layouts']

    # Get gpu kernels

    gpu_weight_proj_kernel = conf['app_state']['gpu']['prepared_kernels'
            ]['weight_proj']
    gpu_proj_to_complex_kernel = conf['app_state']['gpu'
            ]['prepared_kernels']['proj_to_complex']
    gpu_filter_proj_kernel = conf['app_state']['gpu']['prepared_kernels'
            ]['filter_proj']
    gpu_complex_to_proj_kernel = conf['app_state']['gpu'
            ]['prepared_kernels']['complex_to_proj']
    gpu_generate_volume_weight_kernel = conf['app_state']['gpu'
            ]['prepared_kernels']['generate_volume_weight']
    gpu_backproject_kernel = conf['app_state']['gpu']['prepared_kernels'
            ]['backproject']

    # Get GPU data structures

    gpu_projs_data = get_cu_data(conf, 'projs_data')
    gpu_complex_proj = get_cu_data(conf, 'complex_proj')
    gpu_proj_weight_matrix = get_cu_data(conf, 'proj_weight_matrix')
    gpu_proj_filter_array = get_cu_data(conf, 'proj_filter_array')
    gpu_volume_weight_matrix = get_cu_data(conf, 'volume_weight_matrix')
    gpu_transform_matrix = get_cu_data(conf, 'transform_matrix')
    gpu_combined_matrix = get_cu_data(conf, 'combined_matrix')
    gpu_recon_chunk = get_cu_data(conf, 'recon_chunk')
    gpu_z_voxel_coordinates = get_cu_data(conf, 'z_voxel_coordinates')

    # Get recon chunk index

    chunk_index = conf['app_state']['chunk']['idx']

    # Get projection index

    proj_index = conf['app_state']['backproject']['proj_idx']

    # Get projection angle

    proj_angle_rad = radians(fdt(proj['angle']))

    # Get projection row offset

    proj_row_offset = uint32(conf['app_state']['projs']['boundingbox'
                             ][0, 0])
    nr_proj_rows = uint32(conf['app_state']['projs']['boundingbox'][0,
                          1] - proj_row_offset)

    # Offset gpu projection data according to current projection

    proj_chunk_index = proj_index - conf['app_state']['projs']['first']
    gpu_proj_offset = proj_chunk_index * gpu_projs_data.shape[1] \
        * gpu_projs_data.shape[2]
    gpu_proj_shape = (gpu_projs_data.shape[1], gpu_projs_data.shape[2])
    gpu_proj_data = gpu_array_alloc_offset(gpu_projs_data,
            gpu_proj_offset, gpu_proj_shape)

    if proj['filtered']:
        pass
    else:

        # Start filter timer
        # NOTE: The weighting is a part of projection filtering

        # Weight projection, this transforms projection
        # from data type 'fdt' to complex64.
        # If weighting is skipped, the transformation still needs to be done

        if conf['proj_weight'] != 'skip':
            timelog.set(conf, 'verbose', 'proj_weight', barrier=True)
            weight_proj(
                gpu_proj_data,
                proj_row_offset,
                gpu_proj_weight_matrix,
                gpu_weight_proj_kernel,
                gpu_stream,
                gpu_layouts,
                )
            timelog.log(conf, 'verbose', 'proj_weight', barrier=True)

        if conf['checksum']:
            chunk_view = gpu_proj_data.get().ravel()
            log_checksum('Weighted projs chunk', chunk_view,
                         chunk_view.size)

        # Prefilter projection on GPU

        if conf['proj_filter'] != 'skip':
            timelog.set(conf, 'verbose', 'proj_filter', barrier=True)

            # Get fft plan

            fft_plan = conf['app_state']['gpu']['fft_plan']

            # Transform projection from float to complex

            proj_to_complex(
                gpu_complex_proj,
                gpu_proj_data,
                nr_proj_rows,
                gpu_proj_to_complex_kernel,
                gpu_stream,
                gpu_layouts,
                )

            fft_plan.execute(gpu_complex_proj, data_out=None,
                             inverse=False, batch=int(nr_proj_rows),
                             wait_for_finish=None)

            filter_proj(gpu_complex_proj, gpu_proj_filter_array,
                        gpu_filter_proj_kernel, gpu_stream, gpu_layouts)

            fft_plan.execute(gpu_complex_proj, data_out=None,
                             inverse=True, batch=int(nr_proj_rows),
                             wait_for_finish=None)

            # Transform projection from complex to float

            complex_to_proj(gpu_proj_data, gpu_complex_proj,
                            gpu_complex_to_proj_kernel, gpu_stream,
                            gpu_layouts)

            timelog.log(conf, 'verbose', 'proj_filter', barrier=True)

            if conf['checksum']:
                chunk_view = gpu_proj_data.get().ravel()
                log_checksum('Filtered projs chunk', chunk_view,
                             chunk_view.size)

        # Save filtered projection if requested

        if conf['save_filtered_projs_data_path']:
            logging.debug('Saving filtered projection data')

            timelog.set(conf, 'verbose', 'proj_save')
            filtered_proj_data = real(gpu_complex_proj.get())[:, :
                    conf['detector_columns']]

            fd = open(conf['save_filtered_projs_data_path'], 'r+b', 0)
            fd.seek(fdt(0).nbytes * proj['index']
                    * filtered_proj_data.shape[0]
                    * filtered_proj_data.shape[1])
            save_auto(fd, filtered_proj_data)
            fd.close()

            timelog.log(conf, 'verbose', 'proj_save')

    # Generate tranform matrix based on projection angle
    # and move it to GPU

    timelog.set(conf, 'verbose', 'transform_matrix', barrier=True)
    gpu_transform_matrix.set(generate_transform_matrix(
        proj_angle_rad,
        conf['detector_pixel_width'],
        conf['detector_pixel_height'],
        conf['detector_column_shift'],
        conf['detector_row_shift'],
        conf['source_distance'],
        conf['detector_distance'],
        conf['detector_shape'],
        fdt,
        ))
    timelog.log(conf, 'verbose', 'transform_matrix', barrier=True)

    # If volume weight not given in conf, auto generate it

    if conf['volume_weight'] != 'skip':
        timelog.set(conf, 'verbose', 'volume_weight', barrier=True)
        if conf['volume_weight']:
            gpu_volume_weight_matrix.set(get_npy_data(conf,
                    'volume_weight_matrix')[proj_index])
        else:
            generate_volume_weight(
                gpu_volume_weight_matrix,
                gpu_combined_matrix,
                proj_angle_rad,
                gpu_generate_volume_weight_kernel,
                gpu_stream,
                gpu_layouts,
                )

        timelog.log(conf, 'verbose', 'volume_weight', barrier=True)

    timelog.set(conf, 'verbose', 'backproject', barrier=True)
    backproject_proj(
        gpu_recon_chunk,
        gpu_proj_data,
        proj_row_offset,
        chunk_index,
        gpu_z_voxel_coordinates,
        gpu_transform_matrix,
        gpu_combined_matrix,
        gpu_volume_weight_matrix,
        gpu_backproject_kernel,
        gpu_stream,
        gpu_layouts,
        )

    timelog.log(conf, 'verbose', 'backproject', barrier=True)

    return conf


