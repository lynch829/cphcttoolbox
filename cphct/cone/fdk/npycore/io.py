#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - NumPy core specific input/ouput helpers
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

"""NumPy core specific input/output helper functions"""

import os

from cphct.conf import int_pow2_value
from cphct.io import expand_path
from cphct.misc import nextpow2
from cphct.npycore import fromfile, tan, zeros, pi
from cphct.npycore.io import npy_alloc
from cphct.npycore.utils import supported_proj_filters, \
    generate_proj_filter
from cphct.cone.fdk.npycore.kernels import generate_proj_weight_matrix
from cphct.cone.fdk.io import fill_fdk_conf


def __set_proj_weight_matrix(conf, fdt):
    """
    Set projection weight matrix array used when weighting projections
    prior to filtering. 

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : conf
       Configuration dictionary.
    
    Raises
    ------
    ValueError
       If conf['proj_weight'] is set but is neither a valid float value
       nor a valid projection weight file
    """

    proj_weight_path = expand_path(conf['working_directory'],
                                   conf['proj_weight'])

    if conf['proj_weight'] != 'skip':

        matrix_shape = (conf['detector_rows'], conf['detector_columns'])
        proj_weight_matrix = zeros(matrix_shape, dtype=fdt)

        if not conf['proj_weight']:

            proj_weight_matrix[:] = generate_proj_weight_matrix(
                conf['detector_rows'],
                conf['detector_columns'],
                conf['detector_row_shift'],
                conf['detector_column_shift'],
                conf['detector_pixel_height'],
                conf['detector_pixel_width'],
                conf['source_distance'],
                conf['detector_distance'],
                conf['detector_shape'],
                fdt,
                )
        elif os.path.isfile(proj_weight_path):
            try:
                tmp = fromfile(proj_weight_path, dtype=fdt)
                tmp.shape = matrix_shape
                proj_weight_matrix[:] = tmp
            except Exception:
                msg = 'Invalid projection weight file: \'%s\' ' \
                    % proj_weight_path
                raise ValueError(msg)
        else:
            try:
                proj_weight_val = fdt(conf['proj_weight'])
            except:
                msg = 'projection_weight: \'%s\', ' % conf['proj_weight'
                        ] \
                    + 'must be either a valid filepath or a float value'
                raise ValueError(msg)

            proj_weight_matrix[:] = proj_weight_val

        npy_alloc(conf, 'proj_weight_matrix', proj_weight_matrix)

    return conf


def __update_proj_filter_width(conf, filter_width=None):
    """
    Create projection filter width if not provided by user or set in *filter_width*
    Autogenerated filter length is twice the conf['detector_columns'] ceiled
    to the next power of 2 as fft operates on arrays with length of power 2.
    Minimum autogenerated projection filter width is 64.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    filter_width : int, optional
        New projection filter width value

    Returns
    -------
    output : conf
       Configuration dictionary.
       
    """

    if filter_width:
        conf['proj_filter_width'] = int_pow2_value(filter_width)
    elif conf['proj_filter_width'] == -1:
        conf['proj_filter_width'] = max(64, nextpow2(2
                * conf['detector_columns']))

    return conf


def __set_proj_filter_array(conf, fdt):
    """
    Set projection filter array used when filtering projections
    prior to reconstruction.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : conf
       Configuration dictionary.
    
    Raises
    ------
    ValueError
       If conf['proj_filter'] is neither a filepath 
       nor in the list of supported filters
       Or if conf['proj_filter_width'] is less than conf['detector_columns']
    """

    proj_filter_path = expand_path(conf['working_directory'],
                                   conf['proj_filter'])

    if conf['proj_filter'] != 'skip':
        if os.path.isfile(proj_filter_path):
            try:
                proj_filter_array = fromfile(proj_filter_path,
                        dtype=fdt)
                __update_proj_filter_width(conf, len(proj_filter_array))
            except Exception:
                msg = 'Invalid projection filter file: \'%s\' ' \
                    % proj_filter_path
                raise ValueError(msg)
        elif conf['proj_filter'] in supported_proj_filters("fdk"):
            __update_proj_filter_width(conf)
            proj_filter_array = generate_proj_filter(conf['proj_filter'
                    ], conf['proj_filter_width'],
                    conf['proj_filter_scale'],
                    conf['proj_filter_nyquist_fraction'], fdt)
        else:
            msg = 'proj_filter: \'%s\' is neither a filepath ' \
                % conf['proj_filter'] \
                + 'nor in allowed projection filters: %s' \
                % supported_proj_filters("fdk")
            raise ValueError(msg)

        if conf['proj_filter_width'] < conf['detector_columns']:
            msg = 'Projection filter width to small: %s, ' \
                % conf['proj_filter_width'] \
                + 'must be >= the number of detector columns: %s' \
                % conf['detector_columns']
            raise ValueError(msg)

        npy_alloc(conf, 'proj_filter_array', proj_filter_array)
    else:

        # Still set helper variables for other functions to use

        __update_proj_filter_width(conf)

    return conf


def __set_volume_weight_matrix(conf, fdt):
    """
    Set volume weight matrix used when weighting 
    the reconstructed volume in the x,y plane

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : conf
       Configuration dictionary.
    
    Raises
    ------
    ValueError
       If conf['volume_weight'] is set but is neither a valid float value
       nor a valid volume weight file
    """

    volume_weight_path = expand_path(conf['working_directory'],
            conf['volume_weight'])

    if conf['volume_weight'] and conf['volume_weight'] != 'skip':

        # NOTE: This may be changed to a 2D matrix generated
        #       from the FDK,CU,CL 'reconstruct_proj' for each projection
        #       in order to save memory

        matrix_shape = (conf['projs_per_turn'], conf['y_voxels'],
                        conf['x_voxels'])
        volume_weight_matrix = zeros(matrix_shape, dtype=fdt)

        if os.path.isfile(volume_weight_path):
            try:
                tmp = fromfile(volume_weight_path, dtype=fdt)
                tmp.shape = matrix_shape

                volume_weight_matrix[:] = tmp
            except Exception:
                msg = 'Invalid volume weight file: \'%s\' ' \
                    % volume_weight_path
                raise ValueError(msg)
        else:
            try:
                volume_weight_val = fdt(conf['volume_weight'])
            except:
                msg = 'volume_weight: \'%s\', ' % conf['volume_weight'] \
                    + 'must be either a valid filepath or a float value'
                raise ValueError(msg)

            volume_weight_matrix[:] = volume_weight_val

        npy_alloc(conf, 'volume_weight_matrix', volume_weight_matrix)

    return conf


def fill_fdk_npycore_conf(conf):
    """Remaining configuration after handling command line options.
    Casts all floating point results using float data type from conf.
    This version is for the shared NumPy core.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary filled with NumPy core settings.
    """

    fill_fdk_conf(conf)

    fdt = conf['data_type']

    # Make sure all float values get the right precision before we continue

    for (key, val) in conf.items():
        if isinstance(val, float):
            conf[key] = fdt(val)

    # Apparently FDK does not include the half pixel shift to compensate for
    # the use of pixel centers. Increment it here instead.

    conf['detector_row_shift'] += 0.5
    conf['detector_column_shift'] += 0.5

    # Auto select optimal detector size if not provided

    if conf['detector_pixel_width'] <= 0.0 \
        or conf['detector_pixel_height'] <= 0.0 or conf['detector_width'
            ] <= 0.0 or conf['detector_height'] <= 0.0:

        # Make sure detector is big enough to avoid truncation along the
        # axes (Tam-Danielsson window).
        # Please refer to 4.2/5.2 in Noo, Pack and Heuscher paper for the
        # details in relation to flat and curved detectors:
        # "Exact helical reconstruction using native cone-beam geometries"
        # detector_half_width is the distance to the outermost pixel center
        #
        # Avoid rounding issues from actually resulting in too small a
        # detector by forcing the window to fit in one fewer rows and columns

        conf['detector_half_width'] = conf['detector_half_height'] = \
            fdt(conf['scan_diameter'] * tan(conf['half_fan_angle']))
        conf['detector_pixel_height'] = fdt(2.0
                * conf['detector_half_height']
                / max(conf['detector_rows'] - 2, 1))
        if conf['detector_shape'] == 'flat':
            conf['detector_pixel_width'] = fdt(2.0
                    * conf['detector_half_width']
                    / max(conf['detector_columns'] - 2, 1))
        elif conf['detector_shape'] == 'curved':
            conf['detector_pixel_width'] = fdt(2.0
                    * conf['scan_diameter'] * conf['half_fan_angle']
                    / max(conf['detector_columns'] - 2, 1))

    # Now map measured curve length to polar angle span if curved

    if conf['detector_shape'] == 'flat':
        conf['detector_pixel_span'] = conf['detector_pixel_width']
    else:
        conf['detector_pixel_span'] = fdt(conf['detector_pixel_width']
                / conf['scan_diameter'])

    # Printable detector dimensions in cm

    conf['detector_pixel_size'] = (conf['detector_pixel_height'],
                                   conf['detector_pixel_width'])
    conf['detector_height'] = fdt(conf['detector_pixel_size'][0]
                                  * conf['detector_rows'])
    conf['detector_width'] = fdt(conf['detector_pixel_size'][1]
                                 * conf['detector_columns'])
    conf['detector_half_width'] = fdt(0.5 * conf['detector_width'])
    conf['detector_size'] = (conf['detector_height'],
                             conf['detector_width'])

    conf['volume_weight_factor'] = fdt(pi / conf['projs_per_turn'])

    if conf['proj_filter_scale'] < 0.0:
        conf['proj_filter_scale'] = conf['detector_pixel_width']

    # Set up additional vars based on final conf

    # Initialize projection weight matrix based on conf settings

    __set_proj_weight_matrix(conf, fdt)

    # Initialize projection filter array based on conf settings

    __set_proj_filter_array(conf, fdt)

    # Initialize reconstructed volume weight matrix based on conf settings

    __set_volume_weight_matrix(conf, fdt)

    return conf


