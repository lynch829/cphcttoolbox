#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# interpolate dead - plugin to interpolate dead pixels with neighbours
# Copyright (C) 2012-2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Interpolate dead pixels plugin to interpolate dead pixel values with,
their neighbours in the Z-dimension
"""

from cphct.npycore.io import load_helper_proj
from cphct.npycore.utils import interpolate_proj_pixels
from cphct.plugins import get_plugin_var

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, dead_pixels):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Set up dead pixel helper array

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    dead_pixels : str
        File path to dead pixel projection, If dead_pixels='dead_pixels' 
        the dead pixels matrix is extracted from shared plugin vars
        
    Raises
    ------
    ValueError
        If provided dead_pixels is neither 'dead_pixels' or
        a suitable projection file
    """

    # Fill dead pixels

    if dead_pixels == 'dead_pixels':
        dead_pixels_matrix = get_plugin_var(conf, 'dead_pixels')
    else:
        dead_pixels_matrix = load_helper_proj(dead_pixels, conf,
                conf['input_data_type'])

    __plugin_state__['dead_pixels'] = dead_pixels_matrix


def plugin_exit(conf, dead_pixels):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Clean up after helper arrays.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    dead_pixels : str
        File path to dead pixel projection, If dead_pixels='dead_pixels' 
        the dead pixels matrix is extracted from shared plugin vars
    """

    __plugin_state__.clear()


def preprocess_input(
    input_data,
    input_meta,
    conf,
    dead_pixels,
    ):
    """Convert measured intensity input values to attenuation values.

    Parameters
    ----------
    input_data : ndarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    dead_pixels : str
        File path to dead pixel projection, If dead_pixels='dead_pixels' 
        the dead pixels matrix is extracted from shared plugin vars

    Returns
    -------
    output : tuple of ndarray and dict
        Returns a 2-tuple of the array of stacked projections and input_meta.
    """

    # Retrieve initialized dead pixel matrix

    dead_pixels = __plugin_state__['dead_pixels']

    # Raise error if input is not a numpy array

    if not hasattr(input_data, 'dtype'):
        raise ValueError('invalid interpolatedead preprocess input array'
                         )

    return (interpolate_proj_pixels(input_data, dead_pixels,
            (conf['detector_rows'], conf['detector_columns']),
            out=input_data), input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
