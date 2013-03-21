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

from cphct.npycore import cos, sqrt, zeros, ones, vstack
from cphct.npycore.io import npy_alloc, get_npy_data
from cphct.npycore import meshgrid
from cphct.npycore.misc import linear_coordinates


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


def generate_proj_weight_matrix(
    detector_rows,
    detector_columns,
    detector_row_shift,
    detector_column_shift,
    detector_pixel_height,
    detector_pixel_width,
    source_distance,
    detector_distance,
    detector_shape,
    fdt,
    ):
    """Calculate the cone-beam flat or curved projection weight

    Parameters
    ----------
    detector_rows : int
       Number of pixel rows in projections
    detector_columns : int
       Number of pixel columns in projections
    detector_row_shift : float
       Center ray aligned pixel row shift
    detector_column_shift : float
       Center ray aligned pixel column shift
    detector_pixel_height : float
       Detector pixel height in cm
    detector_pixel_width : float
       Detector pixel width in cm
    source_distance : float
       Distance in cm from source to isocenter
    detector_distance : float
       Distance in cm from isocenter to detector
    detector_shape : str
       Shape of detector
    fdt : dtype
       Output proj_weight data type.

    Returns
    -------
    output : ndarray
        Returns a proj_weight matrix of *detector_rows* by *detector_columns*
        with dtype *fdt*.
    """

    # From Henrik Turbell's Ph.D thesis:
    #    link: http://www2.cvl.isy.liu.se/ScOut/Theses/PaperInfo/turbell01.html
    #    link: http://www2.cvl.isy.liu.se/Research/Tomo/Turbell/abstract.html
    #
    #    Bibtex:
    #       @PhdThesis{turbell01,
    #       author  = {Henrik Turbell},
    #       title   = {Cone-Beam Reconstruction Using Filtered Backprojection},
    #       school  = {Link{\"o}ping University, Sweden},
    #       year    = {2001},
    #       month   = {February},
    #       address = {SE-581 83 Link\"oping, Sweden},
    #       node    = {Dissertation No. 672, ISBN 91-7219-919-9}
    #       }

    source_detector_distance = source_distance + detector_distance

    # Detector pixel center coordinates in both directions, using coordinate
    # system centered in the middle of the detector.

    cols = (-detector_column_shift
            + linear_coordinates(-detector_columns / 2.0,
            detector_columns / 2.0, detector_columns, True, fdt)) \
        * detector_pixel_width

    rows = (-detector_row_shift + linear_coordinates(-detector_rows
            / 2.0, detector_rows / 2.0, detector_rows, True, fdt)) \
        * detector_pixel_height

    if detector_shape == 'curved':

        # Part of equation 3.9 Turbell

        col_rads = abs(cols / source_detector_distance)
        col_weight = cos(col_rads)

        (col_weight_grid, row_grid) = meshgrid(col_weight, rows)

        row_weight_grid = source_detector_distance \
            / sqrt(source_detector_distance ** 2 + row_grid ** 2)

        proj_weight_matrix = col_weight_grid * row_weight_grid
    else:

        # Equation 3.5 Turbell

        (col_grid, row_grid) = meshgrid(cols, rows)
        proj_weight_matrix = source_detector_distance \
            / sqrt(source_detector_distance ** 2 + row_grid ** 2
                   + col_grid ** 2)

    return proj_weight_matrix


def generate_combined_matrix(
    x_min,
    x_max,
    x_voxels,
    y_min,
    y_max,
    y_voxels,
    fdt,
    ):
    """
    Generates a matrix containing the coordinates of all
    x and y voxel combinations for a single z voxel coordinate.
    Additional space is allocated for a dummy dimension needed
    by dot products performed during reconstruction.

    Parameters
    ----------
    x_min: float
       Field of View minimum x coordinate in cm.
    x_max: float
       Field of View maximum x coordinate in cm.
    x_voxels: int
       Field of View resolution in x.
    y_min: float
       Field of View minimum y coordinate in cm.
    y_max: float
       Field of View maximum y coordinate in cm.
    y_voxels: int
       Field of View resolution in y.
    fdt : type
        Output filter data type.

    Returns
    -------
    output : ndarray
        Returns a combination of all *x_voxels* by *y_voxels* coordinate
        positions with dtype *fdt*.
    """

    x_coords = linear_coordinates(x_min, x_max, x_voxels, True, fdt)
    y_coords = linear_coordinates(y_min, y_max, y_voxels, True, fdt)

    (y_coords_grid, x_coords_grid) = meshgrid(y_coords, x_coords)
    (flat_y_coords_grid, flat_x_coords_grid) = \
        (fdt(y_coords_grid.flatten('F')), fdt(x_coords_grid.flatten('F'
         )))

    # Allocate space for one slice, the z values are filled in
    # when looping over slices in the reconstruction kernel

    flat_z_coords_grid = zeros(len(flat_x_coords_grid), dtype=fdt)

    # Dummy must be ones in order to get correct dot products
    # in the reconstruction kernel

    dummy = ones(len(flat_x_coords_grid), dtype=fdt)

    combined_matrix = vstack([flat_x_coords_grid, flat_y_coords_grid,
                             flat_z_coords_grid, dummy])

    return combined_matrix


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


