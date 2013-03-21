#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# conf - shared fan beam configuration helpers
# Copyright (C) 2011  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Shared fan beam configuration helper functions"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Note: this is ugly but we need to inherit and re-expose all functions from
# parent module

from cphct.conf import *


def _shared_opts():
    """Shared fan beam options for all engines

    Returns
    -------
    output : dict
        Returns a dictionary of options helper dictionaries.
    """

    opts = {'filter': {
        'long': 'filter',
        'short': None,
        'args': str,
        'handler': str_value,
        'default': 'none',
        'description': 'Type of projection filter to apply',
        }}

    return opts


def _npy_opts():
    """Fan beam options for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of numpy specific options helper dictionaries.
    """

    opts = {}
    return opts


def _cu_opts():
    """Fan beam options for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of cuda specific options helper dictionaries.
    """

    opts = {}
    return opts


def default_fan_opts():
    """Engine independent options

    Returns
    -------
    output : dict
        Returns a dictionary of fan options helper dictionaries.
    """

    opts = default_base_opts()
    opts.update(_shared_opts())
    return opts


def default_fan_npy_opts():
    """Numpy specific options

    Returns
    -------
    output : dict
        Returns a dictionary of numpy specific fan options helper
        dictionaries.
    """

    opts = default_base_npy_opts()
    opts.update(_shared_opts())
    opts.update(_npy_opts())
    return opts


def default_fan_cu_opts():
    """Cuda specific options

    Returns
    -------
    output : dict
        Returns a dictionary of cuda specific fan options helper dictionaries.
    """

    opts = default_base_cu_opts()
    opts.update(_shared_opts())
    opts.update(_cu_opts())
    return opts


def default_fan_conf():
    """Engine independent configuration dictionary with default values

    Returns
    -------
    output : dict
        Returns a dictionary of base conf settings.
    """

    conf = {}
    for (key, val) in default_fan_opts().items():
        conf[key] = val['default']
    return conf


def default_fan_npy_conf():
    """Configuration dictionary with default values for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of base numpy conf settings.
    """

    conf = {}
    for (key, val) in default_fan_npy_opts().items():
        conf[key] = val['default']
    return conf


def default_fan_cu_conf():
    """Configuration dictionary with default values for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of base cuda conf settings.
    """

    conf = {}
    for (key, val) in default_fan_cu_opts().items():
        conf[key] = val['default']
    return conf
