#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# hounsfield - hounsfield plugin to scale voxel data to hounsfield units (HU)
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

"""hounsfield plugin to scale voxel data to hounsfield units (HU)"""

from cphct.npycore.utils import hounsfield_scale
from cphct.plugins import get_plugin_var

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, raw_voxel_water):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Just check args in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    raw_voxel_water : float
        The raw voxel value of distilled water

    Raises
    ------
    ValueError
        If provided raw_voxel_water is neither 'raw_voxel_water' 
        nor a valid floating point number.
    """

    __plugin_state__['name'] = __name__

    if raw_voxel_water == 'raw_voxel_water':
        raw_voxel_water = get_plugin_var(conf, 'raw_voxel_water')
    else:
        raw_voxel_water = float(raw_voxel_water)

    __plugin_state__['raw_voxel_water'] = raw_voxel_water


def plugin_exit(conf, raw_voxel_water):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Nothing to do in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    raw_voxel_water : float
        The raw voxel value of distilled water
    """

    __plugin_state__.clear()


def postprocess_output(
    output_data,
    output_meta,
    conf,
    raw_voxel_water,
    ):
    """Convert reconstructed data to the hounsfield scale 
    based on the raw voxel value of distilled water.

    Parameters
    ----------
    output_data : ndarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    raw_voxel_water : float
        The raw voxel value of distilled water

    Returns
    -------
    output : tuple of ndarray and dict
        Returns a tuple of the data array scaled to hounsfield units
        and meta list.
    """

    raw_voxel_water = __plugin_state__['raw_voxel_water']

    # Raise error if input is not a numpy array

    if not hasattr(output_data, 'dtype'):
        raise ValueError('invalid hounsfield postprocess input array')

    return (hounsfield_scale(output_data, raw_voxel_water,
            out=output_data), output_meta)


if __name__ == '__main__':
    print 'no unit tests!'

