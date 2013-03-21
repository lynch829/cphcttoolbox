#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# normalize - normalize plugin to scale data values to full range
# Copyright (C) 2012  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Normalize plugin to scale input and output data to full valid range"""

from cphct.npycore import arange, float32, float64, float128, uint16, \
    uint32, uint64, int16, int32, int64
from cphct.npycore.utils import normalize_array

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, normalize_min=None, normalize_max=None):
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
    normalize_min : float or None
        The minimum value to normalize input array to.
    normalize_max : float or None
        The maximum value to normalize input array to.

    Raises
    ------
    ValueError
        If provided normalize_min or normalize_max is not None or a valid
        floating point number
        or if normalize_min is greater than normalize_max.
    """

    __plugin_state__['name'] = __name__
    if normalize_min is None:
        normalize_min = 0
    if normalize_max is None:
        normalize_max = 65535
    min_val = float(normalize_min)
    max_val = float(normalize_max)
    if min_val > max_val:
        raise ValueError('normalize_min is greater than normalize_max')


def plugin_exit(conf, normalize_min=None, normalize_max=None):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Nothing to do in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    normalize_min : float or None
        The minimum value to normalize input array to.
    normalize_max : float or None
        The maximum value to normalize input array to.
    """

    __plugin_state__.clear()


def preprocess_input(
    input_data,
    input_meta,
    conf,
    normalize_min=None,
    normalize_max=None,
    ):
    """Normalize input using dtype for range information

    Parameters
    ----------
    input_data : ndarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    normalize_min : float or None
        The minimum value to normalize input array to.
    normalize_max : float or None
        The maximum value to normalize input array to.

    Returns
    -------
    output : tuple of ndarray and list
        Returns a tuple of the data array limited to
        [normalize_min:normalize_max] range and meta list.
    """

    # Raise error if input is not a numpy array

    if not hasattr(input_data, 'dtype'):
        raise ValueError('invalid preprocess input without dtype attribute'
                         )

    # Optional user supplied normalize range with uint16 fall back

    if normalize_min is None:
        normalize_min = 0
    if normalize_max is None:
        normalize_max = 65535

    return (normalize_array(input_data, normalize_min, normalize_max,
            out=input_data), input_meta)


def postprocess_output(
    output_data,
    output_meta,
    conf,
    normalize_min=None,
    normalize_max=None,
    ):
    """Normalize output using conf type for range information

    Parameters
    ----------
    output_data : ndarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    normalize_min : float or None
        The minimum value to normalize input array to.
    normalize_max : float or None
        The maximum value to normalize input array to.

    Returns
    -------
    output : tuple of ndarray and dict
        Returns a tuple of the data array limited to
        [normalize_min:normalize_max] range and meta list.
    """

    # Raise error if input is not a numpy array

    if not hasattr(output_data, 'dtype'):
        raise ValueError('invalid postprocess input without dtype attribute'
                         )

    # Optional user supplied normalize range with uint16 fall back

    if normalize_min is None:
        normalize_min = 0
    if normalize_max is None:
        normalize_max = 65535

    return (normalize_array(output_data, normalize_min, normalize_max,
            out=output_data), output_meta)


if __name__ == '__main__':
    for data_type in [
        float32,
        float64,
        float128,
        uint16,
        uint32,
        uint64,
        int16,
        int32,
        int64,
        ]:
        data = arange(3, 8, dtype=data_type)
        print 'Normalize data %s (%s)' % (data, data.dtype.name)
        out = normalize_array(data)
        print 'Normalized to %s (%s)' % (out, out.dtype.name)
        data = arange(3, 8, dtype=data_type)
        out = normalize_array(data, 0, 255)
        print 'Normalized to byte range %s (%s)' % (out, out.dtype.name)
        data = arange(-3, 7, dtype=data_type)
        print 'Normalize data %s (%s)' % (data, data.dtype.name)
        out = normalize_array(data)
        print 'Normalized to %s (%s)' % (out, out.dtype.name)
        data = arange(-3, 7, dtype=data_type)
        out = normalize_array(data, 0, 255)
        print 'Normalized to byte range %s (%s)' % (out, out.dtype.name)
