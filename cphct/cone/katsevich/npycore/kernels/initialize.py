#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# initialize - numpy specific initialization helpers
# Copyright (C) 2011-2013  The Cph CT Toolbox Project lead by Brian Vinter
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
from cphct.npycore import pi, zeros, ones, arange, sin, cos, tan, \
    arctan, int32
from cphct.npycore.io import npy_alloc, get_npy_data
from cphct.npycore.misc import linear_coordinates


def init_recon(conf, fdt):
    """
    Initialize data structures for numpy Katsevich reconstruction

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    fdt : dtype
        Float data type (internal precision).

    Returns
    -------
    output : dict
       Returns configuration dictionary filled with numpy Katsevich
       data structures
    """

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
    # system centered in the middle of the detector.
    # We include one extra pixel coordinate in ext version for use in
    # interpolation later.
    # Please note that the row coordinate formula in (82) of the Noo paper
    # is wrong. It should be Nrows instead of Ncols and fig 4 indicates that
    # the pixel center coordinates are used which means both rows and columns
    # should include the '-1' from shift to get the half pixel offset like for
    # the curved detector.

    npy_alloc(conf, 'row_coords_ext', conf['detector_pixel_height']
              * (arange(conf['detector_rows'] + 1, dtype=fdt)
              - conf['detector_row_shift']))
    row_coords_ext = get_npy_data(conf, 'row_coords_ext')
    row_coords = row_coords_ext[:-1]

    npy_alloc(conf, 'col_coords_ext', conf['detector_pixel_span']
              * (arange(conf['detector_columns'] + 1, dtype=fdt)
              - conf['detector_column_shift']))
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
        proj_row_mins = conf['proj_row_mins'] = conf['progress_per_turn'
                ] / pi * min_help / (1 - cos(min_help))
        proj_row_maxs = conf['proj_row_maxs'] = conf['progress_per_turn'
                ] / pi * max_help / (1 - cos(max_help))
    elif conf['detector_shape'] == 'curved':
        proj_row_mins = conf['proj_row_mins'] = \
            -conf['progress_per_turn'] / pi * (pi / 2 + col_coords_ext) \
            / cos(col_coords_ext)
        proj_row_maxs = conf['proj_row_maxs'] = conf['progress_per_turn'
                ] / pi * (pi / 2 - col_coords_ext) / cos(col_coords_ext)
    npy_alloc(conf, 'proj_row_mins', proj_row_mins)
    npy_alloc(conf, 'proj_row_maxs', proj_row_maxs)

    # Matrices for storage in host memory - init with zeros to make sure they
    # are fully initialized for mem size checks

    npy_alloc(conf, 'filter_in', zeros((conf['filter_in_projs'],
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))

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

    input_buffer_projs = (2 + conf['chunk_projs']
                          / conf['projs_per_turn']) \
        * conf['projs_per_turn']
    npy_alloc(conf, 'input_buffer', zeros((input_buffer_projs,
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))
    npy_alloc(conf, 'input_chunk', zeros((conf['chunk_projs'],
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))
    npy_alloc(conf, 'output_chunk', zeros((conf['x_voxels'],
              conf['y_voxels'], conf['chunk_size']), dtype=fdt))


    return conf
