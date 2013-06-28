#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# conf - shared fdk configuration helpers
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

"""Shared FDK configuration helper functions"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Note: this is ugly but we need to inherit and re-expose all functions from
# parent module

from cphct.cone.conf import *
from cphct.io import path_expander
from cphct.npycore.utils import supported_proj_filters

# Valid values for string options:
# In case of different public and internal names we use a dictionary mapping
# the public name to the internal name. Otherwise we just use a shared list of
# strings.

def _shared_opts():
    """Shared FDK options for all engines

    Returns
    -------
    output : dict
        Returns a dictionary of options helper dictionaries.
    """

    opts = {
        'proj_filter': {
            'long': 'proj-filter',
            'short': None,
            'args': str,
            'handler': str,
            'default': 'hamming',
            'description': 'Projection filter filepath ' \
                + 'or one of the builtin filters:\n\t%s' \
                % supported_proj_filters,
            },
        'proj_filter_width': {
            'long': 'proj-filter-width',
            'short': None,
            'args': int,
            'handler': int_pow2_value,
            'default': -1,
            'description': 'FDK projection filter resolution, ' \
                + 'must be a power of two.',
            },
        'proj_filter_scale': {
            'long': 'proj-filter-scale',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1.0,
            'description': 'FDK projection filter scale ' \
                + '(used when generating builtin filters)',
            },
        'proj_filter_nyquist_fraction': {
            'long': 'proj-filter-nyquist-fraction',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 1.0,
            'description': 'FDK projection filter nyquest fraction ' \
                + '(used when generating builtin filters)',
            },
        'proj_weight': {
            'long': 'proj-weight',
            'short': None,
            'args': str,
            'handler': str_value,
            'default': '',
            'description': 'FDK projection weight, filepath or float.',
            },
        'volume_weight': {
            'long': 'volume-weight',
            'short': None,
            'args': str,
            'handler': str_value,
            'default': '',
            'description': 'FDK reconstructed volume weight, ' \
                + 'filepath or float.',
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

    opts = {
        'proj_chunk_size': {
        'long': 'proj-chunk-size',
        'short': None,
        'args': int,
        'handler': int_value,
        'default': 1,
        'description': 'CUDA FDK number of projections processed at a time.',
        },
    }
    return opts


def default_fdk_opts():
    """Engine independent options

    Returns
    -------
    output : dict
        Returns a dictionary of fdk options helper dictionaries.
    """

    opts = default_cone_opts()
    opts.update(_shared_opts())
    return opts


def default_fdk_npy_opts():
    """Numpy specific options

    Returns
    -------
    output : dict
        Returns a dictionary of fdk numpy options helper dictionaries.
    """

    opts = default_cone_npy_opts()
    opts.update(_shared_opts())
    opts.update(_npy_opts())
    return opts


def default_fdk_cu_opts():
    """Cuda specific options

    Returns
    -------
    output : dict
        Returns a dictionary of fdk cuda options helper dictionaries.
    """

    opts = default_cone_cu_opts()
    opts.update(_shared_opts())
    opts.update(_cu_opts())
    return opts


def default_fdk_conf():
    """Engine independent configuration dictionary with default values

    Returns
    -------
    output : dict
        Returns a dictionary of fdk conf settings.
    """

    conf = {}
    for (key, val) in default_fdk_opts().items():
        conf[key] = val['default']
    return conf


def default_fdk_npy_conf():
    """Configuration dictionary with default values for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of fdk numpy conf settings.
    """

    conf = {}
    for (key, val) in default_fdk_npy_opts().items():
        conf[key] = val['default']
    return conf


def default_fdk_cu_conf():
    """Configuration dictionary with default values for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of fdk cuda conf settings.
    """

    conf = {}
    for (key, val) in default_fdk_cu_opts().items():
        conf[key] = val['default']
    return conf


