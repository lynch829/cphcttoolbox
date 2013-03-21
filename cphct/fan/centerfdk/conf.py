#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# conf - shared center slice fdk configuration helpers
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

"""Shared center slice FDK configuration helper functions"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Note: this is ugly but we need to inherit and re-expose all functions from
# parent module

from cphct.fan.conf import *
from cphct.cone.fdk.conf import default_fdk_npy_opts, default_fdk_cu_opts


def _shared_opts():
    """Shared Center FDK options for all engines

    Returns
    -------
    output : dict
        Returns a dictionary of options helper dictionaries.
    """

    # Hard code single center slice setup

    # detector_height, z_min and z_max are arbitrary (unused) values

    opts = {
        'detector_height': {
            'long': None,
            'short': None,
            'args': float,
            'handler': None,
            'default': -1.0,
            'description': 'Detector height in cm: -1 for auto',
            },
        'detector_rows': {
            'long': None,
            'short': None,
            'args': int,
            'handler': None,
            'default': 1,
            'description': 'Number of pixel rows in projections',
            },
        'z_min': {
            'long': None,
            'short': None,
            'args': float,
            'handler': None,
            'default': -1.0,
            'description': 'Field of View minimum z coordinate',
            },
        'z_max': {
            'long': None,
            'short': None,
            'args': float,
            'handler': None,
            'default': 1.0,
            'description': 'Field of View maximum z coordinate',
            },
        'z_voxels': {
            'long': None,
            'short': None,
            'args': int,
            'handler': None,
            'default': 1,
            'description': 'Field of View resolution in z',
            },
        }

    return opts


def _npy_opts():
    """FDK options for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of numpy specific options helper dictionaries.
    """

    opts = {}
    return opts


def _cu_opts():
    """FDK options for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of cuda specific options helper dictionaries.
    """

    opts = {}
    return opts


def default_centerfdk_opts():
    """Engine independent options

    Returns
    -------
    output : dict
        Returns a dictionary of centerfdk options helper dictionaries.
    """

    opts = default_fan_opts()
    opts.update(_shared_opts())
    return opts


def default_centerfdk_npy_opts():
    """Numpy specific options

    Returns
    -------
    output : dict
        Returns a dictionary of numpy specific centerfdk options helper
        dictionaries.
    """

    opts = default_fan_npy_opts()

    # Insert cone fdk defaults and override with center slice defaults

    opts.update(default_fdk_npy_opts())
    opts.update(_shared_opts())
    opts.update(_npy_opts())
    return opts


def default_centerfdk_cu_opts():
    """Cuda specific options

    Returns
    -------
    output : dict
        Returns a dictionary of cuda specific centerfdk options helper
        dictionaries.
    """

    opts = default_fan_cu_opts()

    # Insert cone fdk defaults and override with center slice defaults

    opts.update(default_fdk_cu_opts())
    opts.update(_shared_opts())
    opts.update(_cu_opts())
    return opts


def default_centerfdk_conf():
    """Engine independent configuration dictionary with default values

    Returns
    -------
    output : dict
        Returns a dictionary of centerfdk conf settings.
    """

    conf = {}
    for (key, val) in default_centerfdk_opts().items():
        conf[key] = val['default']
    return conf


def default_centerfdk_npy_conf():
    """Configuration dictionary with default values for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of centerfdk numpy conf settings.
    """

    conf = {}
    for (key, val) in default_centerfdk_npy_opts().items():
        conf[key] = val['default']
    return conf


def default_centerfdk_cu_conf():
    """Configuration dictionary with default values for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of centerfdk cuda conf settings.
    """

    conf = {}
    for (key, val) in default_centerfdk_cu_opts().items():
        conf[key] = val['default']
    return conf
