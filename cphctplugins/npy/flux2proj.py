#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# flux2proj - plugin to convert measured intensities to attenuation projections
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

"""Flux to projection plugin to convert measured intensities to actual
attenuation projections.
"""

from cphct.npycore import allowed_data_types
from cphct.npycore.io import load_helper_proj
from cphct.npycore.utils import check_norm, flux_to_proj
from cphct.plugins import get_plugin_var

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(
    conf,
    zero_norm,
    air_norm,
    air_ref_pixel=None,
    dtype_norm='float32',
    ):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Set up air and zero norm helper arrays

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    zero_norm : str
        Background intensity normalization 'zero_norm', value or file path
        If zero_norm='zero_norm' the zero norm matrix is extracted using
        get_plugin_var
    air_norm : str
        Pure air intensity normalization 'air_norm', value or file path.
        If air_norm='air_norm' the air norm matrix is extracted using
        get_plugin_var
    air_ref_pixel : str, optional
        Tuble of pixel posistion (y,x) in projection containing air value
    dtype_norm : str, optional
        Norm martrices dtype

    Raises
    ------
    ValueError
        If provided dtype_norm is not a valid data type,
        if provided zero_norm value is neither 'zero_norm', 
        a suitable projection file nor
        a single value compatible with dtype_norm
        if provided air_norm value is neither 'air_norm',
        a suitable projection file nor
        a single value compatible with dtype_norm,
        if zero norm is greater than air norm or
        if air_ref_pixel is set not a valid (y,x) index
        
    """

    # Transform dtype_norm string to dtype

    dtype_norm = allowed_data_types[dtype_norm]

    # Fill zero and air norm

    if zero_norm == 'zero_norm':
        zero_norm_matrix = get_plugin_var(conf, 'zero_norm')
    else:
        zero_norm_matrix = load_helper_proj(zero_norm, conf, dtype_norm)

    if air_norm == 'air_norm':
        air_norm_matrix = get_plugin_var(conf, 'air_norm')
    else:
        air_norm_matrix = load_helper_proj(air_norm, conf, dtype_norm)

    if air_ref_pixel is not None:

        # Create air_ref_pixel tuple of (int, int)

        air_ref_list = air_ref_pixel.split(',')
        air_ref_pixel = (int(air_ref_list[0].strip('(').strip()),
                         int(air_ref_list[1].strip(')').strip()))

    check_norm(zero_norm_matrix, air_norm_matrix)
    __plugin_state__['zero_norm'] = zero_norm_matrix
    __plugin_state__['air_norm'] = air_norm_matrix
    __plugin_state__['air_ref_pixel'] = air_ref_pixel


def plugin_exit(
    conf,
    zero_norm,
    air_norm,
    air_ref_pixel=None,
    dtype_norm='float32',
    ):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Clean up after helper arrays.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    zero_norm : str
        Dummy argument
    air_norm : str
        Dummy argument
    air_ref_pixel : str, optional
        Dummy argument
    dtype_norm : str, optional
        Dummy argument
    """

    __plugin_state__.clear()


def preprocess_input(
    input_data,
    input_meta,
    conf,
    zero_norm,
    air_norm,
    air_ref_pixel=None,
    dtype_norm='float32',
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
    zero_norm : str
        Dummy argument
    air_norm : str
        Dummy argument
    air_ref_pixel : str, optional
        Dummy argument
    dtype_norm : str, optional
        Dummy argument
    Returns
    -------
    output : tuple of ndarray and dict
        Returns a 2-tuple of the array of stacked projections and input_meta.
    """

    # Retrieve initialized norm matrices

    zero_norm = __plugin_state__['zero_norm']
    air_norm = __plugin_state__['air_norm']
    air_ref_pixel = __plugin_state__['air_ref_pixel']

    # Raise error if input is not a numpy array

    if not hasattr(input_data, 'dtype'):
        raise ValueError('invalid flux_to_proj preprocess input array')

    return (flux_to_proj(input_data, zero_norm, air_norm,
            (conf['detector_rows'], conf['detector_columns'],
            air_ref_pixel), out=input_data), input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
