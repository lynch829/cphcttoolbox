#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# conf - shared cone beam configuration helpers
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

"""Shared cone beam configuration helper functions"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Note: this is ugly but we need to inherit and re-expose all functions from
# parent module

from cphct.conf import *


def _shared_opts():
    """Shared cone beam options for all engines

    Returns
    -------
    output : dict
        Returns a dictionary of options helper dictionaries.
    """

    # Override general chunk-size for a cone-specific description

    opts = {
        'detector_pixel_height': {
            'long': 'detector-pixel-height',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1,
            'description': 'Detector pixel height in cm',
            'value_help': '-1 for auto',
            },
        'detector_height': {
            'long': 'detector-height',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1,
            'description': 'Detector height in cm',
            'value_help': '-1 for auto',
            },
        'detector_rows': {
            'long': 'detector-rows',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 64,
            'description': 'Number of pixel rows in projections',
            },
        'detector_row_offset': {
            'long': 'detector-row-offset',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 0.0,
            'description': 'Center ray alignment offset in pixel rows',
            },
        'z_min': {
            'long': 'z-min',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1.0,
            'description': 'Field of View minimum z coordinate in cm',
            },
        'z_max': {
            'long': 'z-max',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 1.0,
            'description': 'Field of View maximum z coordinate in cm',
            },
        'z_voxels': {
            'long': 'z-voxels',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 512,
            'description': 'Field of View resolution in z',
            },
        'chunk_size': {
            'long': 'chunk-size',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': -1,
            'description': 'Number of z slices in reconstruction chunks',
            },
        }

    return opts


def _npy_opts():
    """Cone beam options for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of numpy specific options helper dictionaries.
    """

    opts = {}
    return opts


def _cu_opts():
    """Cone beam options for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of cuda specific options helper dictionaries.
    """

    opts = {}
    return opts


def default_cone_opts():
    """Engine independent options

    Returns
    -------
    output : dict
        Returns a dictionary of cone options helper dictionaries.
    """

    opts = default_base_opts()
    opts.update(_shared_opts())
    return opts


def default_cone_npy_opts():
    """Numpy specific options

    Returns
    -------
    output : dict
        Returns a dictionary of cone numpy options helper dictionaries.
    """

    opts = default_base_npy_opts()
    opts.update(_shared_opts())
    opts.update(_npy_opts())
    return opts


def default_cone_cu_opts():
    """Cuda specific options

    Returns
    -------
    output : dict
        Returns a dictionary of cone cuda options helper dictionaries.
    """

    opts = default_base_cu_opts()
    opts.update(_shared_opts())
    opts.update(_cu_opts())
    return opts


def default_cone_conf():
    """Engine independent configuration dictionary with default values

    Returns
    -------
    output : dict
        Returns a dictionary of cone conf settings.
    """

    conf = {}
    for (key, val) in default_cone_opts().items():
        conf[key] = val['default']
    return conf


def default_cone_npy_conf():
    """Configuration dictionary with default values for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of cone numpy conf settings.
    """

    conf = {}
    for (key, val) in default_cone_npy_opts().items():
        conf[key] = val['default']
    return conf


def default_cone_cu_conf():
    """Configuration dictionary with default values for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of cone cuda conf settings.
    """

    conf = {}
    for (key, val) in default_cone_cu_opts().items():
        conf[key] = val['default']
    return conf
