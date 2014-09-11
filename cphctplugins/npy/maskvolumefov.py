#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# maskvolumefov - maskvolumefov plugin to mask volume FoV
# Copyright (C) 2012-2014  The Cph CT Toolbox Project lead by Brian Vinter
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

"""maskvolumefov plugin to mask volume FoV"""

from cphct.npycore import zeros, sqrt
from cphct.npycore.misc import linear_coordinates
from cphct.plugins import get_plugin_var

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, fov_radius=None):
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
    fov_radius : float, optional
        The included FoV radius in cm
    """

    __plugin_state__['name'] = __name__

    fdt = conf['data_type']
    x_voxels = conf['x_voxels']
    y_voxels = conf['y_voxels']

    x_min = conf['x_min']
    x_max = conf['x_max']
    y_min = conf['y_min']
    y_max = conf['y_max']

    x_coords_2 = linear_coordinates(x_min, x_max, x_voxels, True, fdt) \
        ** 2
    y_coords_2 = linear_coordinates(y_min, y_max, y_voxels, True, fdt) \
        ** 2

    if fov_radius is None:
        fov_radius = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    fov_radius = fdt(fov_radius)

    fov_mask = zeros((y_voxels, x_voxels), dtype=fdt)

    for y in xrange(y_voxels):
        fov = sqrt(y_coords_2[y] + x_coords_2)
        fov_mask[y, fov <= fov_radius] = 1.0

    __plugin_state__['fov_mask'] = fov_mask


def plugin_exit(conf, fov_radius=None):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Nothing to do in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    fov_radius : int, optional
        The included FoV radius in cm
    """

    __plugin_state__.clear()


def postprocess_output(
    output_data,
    output_meta,
    conf,
    fov_radius=None,
    ):
    """
    Mask reconstructed data to specified FoV radius.

    Parameters
    ----------
    output_data : ndarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    fov_radius : int, optional
        The included FoV radius in cm
    
    Returns
    -------
    output : tuple of ndarray and dict
        Returns a tuple of the data array masked to specified FoV
        radius and meta list.
    """

    # Raise error if input is not a numpy array

    fov_mask = __plugin_state__['fov_mask']

    if not hasattr(output_data, 'dtype'):
        raise ValueError('invalid maskvolumefov postprocess input array'
                         )

    output_data[:] *= fov_mask

    return (output_data, output_meta)


if __name__ == '__main__':
    print 'no unit tests!'

