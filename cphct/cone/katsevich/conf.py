#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# conf - shared katsevich configuration helpers
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

"""Shared Katsevich configuration helper functions"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Note: this is ugly but we need to inherit and re-expose all functions from
# parent module

from cphct.cone.conf import *


def _shared_opts():
    """Shared Katsevich options for all engines

    Returns
    -------
    output : dict
        Returns a dictionary of options helper dictionaries.
    """

    # detector_rebin_rows is the number of rows to rebin to during filtering.
    # We choose to filter with the same number of rows per default - paper
    # recommends a factor between 0.6 and 1.5 of the detector rows for
    # identical pixel width and height. Our own experiments show that it
    # should be at least equal to the number of rows and and up to 1.5 times
    # that number really helps with few detector rows.
    # Noo's paper has some general recommendations in 4.3.2 with an odd
    # number of rebin rows (2*M+1), so that 2M/Nrows ~= 2 for a half fan
    # angle alpha m of 26 degrees

    # TODO: select detector_rebin_rows dynamically within that range unless set

    opts = {'detector_rebin_rows': {
        'long': 'detector-rebin-rows',
        'short': None,
        'args': int,
        'handler': int_value,
        'default': 64,
        'description': 'Number of rows in projection rebinning',
        }, 'progress_per_turn': {
        'long': 'progress-per-turn',
        'short': 'P',
        'args': float,
        'handler': float_value,
        'default': 1.0,
        'description': 'Geometrical helix pitch in cm, i.e. conveyor ' + \
                'progress per turn',
        }}

    return opts


def _npy_opts():
    """Katsevich options for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of numpy specific options helper dictionaries.
    """

    opts = {}
    return opts


def _cu_opts():
    """Katsevich options for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of cuda specific options helper dictionaries.
    """

    opts = {}
    return opts


def default_katsevich_opts():
    """Engine independent options

    Returns
    -------
    output : dict
        Returns a dictionary of katsevich options helper dictionaries.
    """

    opts = default_cone_opts()
    opts.update(_shared_opts())
    return opts


def default_katsevich_npy_opts():
    """Numpy specific options

    Returns
    -------
    output : dict
        Returns a dictionary of katsevich numpy options helper dictionaries.
    """

    opts = default_cone_npy_opts()
    opts.update(_shared_opts())
    opts.update(_npy_opts())
    return opts


def default_katsevich_cu_opts():
    """Cuda specific options

    Returns
    -------
    output : dict
        Returns a dictionary of katsevich cuda options helper dictionaries.
    """

    opts = default_cone_cu_opts()
    opts.update(_shared_opts())
    opts.update(_cu_opts())
    return opts


def default_katsevich_conf():
    """Engine independent configuration dictionary with default values

    Returns
    -------
    output : dict
        Returns a dictionary of katsevich conf settings.
    """

    conf = {}
    for (key, val) in default_katsevich_opts().items():
        conf[key] = val['default']
    return conf


def default_katsevich_npy_conf():
    """Configuration dictionary with default values for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of katsevich numpy conf settings.
    """

    conf = {}
    for (key, val) in default_katsevich_npy_opts().items():
        conf[key] = val['default']
    return conf


def default_katsevich_cu_conf():
    """Configuration dictionary with default values for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of katsevich cuda conf settings.
    """

    conf = {}
    for (key, val) in default_katsevich_cu_opts().items():
        conf[key] = val['default']
    return conf
