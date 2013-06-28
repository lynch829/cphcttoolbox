#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# initialize - numpy specific initialization helpers
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

from cphct.npycore import zeros
from cphct.npycore.io import npy_alloc, get_npy_data
from cphct.npycore.misc import linear_coordinates
from cphct.cone.fdk.npycore.kernels.base import generate_combined_matrix


def __generate_proj_filter_matrix(
    filter_array,
    filter_width,
    filter_height,
    fdt,
    ):
    """
    Generate filter matrix used when filtering the projections
    prior to reconstruction.

    Parameters
    ----------
    filter_array : ndarray
        Array of filter row values.
    filter_width : int
        Filter width in pixels
    filter_height : int
        Filter height in pixels
    fdt : dtype
        Output filter data type.

    Returns
    -------
    output : ndarray
        Returns a filter matrix of *filter_height* by *filter_array.shape*
        with dtype *fdt*.
    """

    filter_matrix = zeros((filter_height, filter_width), dtype=fdt)

    filter_matrix[:] = filter_array

    return filter_matrix


def init_recon(conf, fdt):
    """
    Initialize data structures for numpy FDK reconstruction

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    fdt : dtype
        Float data type (internal precision).

    Returns
    -------
    output : dict
       Returns configuration dictionary filled with numpy FDK reconstruction
       data structures
    """

    # Create proj_filter_matrix

    if conf['proj_filter'] != 'skip':
        npy_alloc(conf, 'proj_filter_matrix',
                  __generate_proj_filter_matrix(get_npy_data(conf,
                  'proj_filter_array'), conf['proj_filter_width'],
                  conf['detector_rows'], fdt))

    # Create combined matrix

    npy_alloc(conf, 'combined_matrix', generate_combined_matrix(
        conf['x_min'],
        conf['x_max'],
        conf['x_voxels'],
        conf['y_min'],
        conf['y_max'],
        conf['y_voxels'],
        fdt,
        ))

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

    return conf


