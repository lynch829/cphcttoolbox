#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# resetproj - plugin to reset all projections to a provided value or template
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

"""Reset projection values plugin useful for debugging"""

from cphct.npycore.io import load_helper_proj
from cphct.plugins import get_plugin_var

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, reset_norm):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Set up reset helper array

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    reset_norm : str
        File path to reset pixel projection or single value. If
        reset_norm='reset_norm' the reset norm pixels matrix is extracted from
        shared plugin vars.

    Raises
    ------
    ValueError
        If provided reset_norm value is neither a suitable projection file nor
        a single value compatible with input data type
    """

    # Fill reset norm

    if reset_norm == 'reset_norm':
        reset_norm_matrix = get_plugin_var(conf, 'reset_norm')
    else:
        reset_norm_matrix = load_helper_proj(reset_norm, conf,
                                             conf['input_data_type'])
    __plugin_state__['reset_norm'] = reset_norm_matrix


def plugin_exit(conf, reset_norm):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Clean up after helper arrays.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    reset_norm : str
        Forced constant normalization value or file path
    """

    __plugin_state__.clear()


def preprocess_input(
    input_data,
    input_meta,
    conf,
    reset_norm,
    ):
    """Force all *input_data* values to *reset_norm* values.

    Parameters
    ----------
    input_data : ndarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    reset_norm : str
        Forced constant normalization value or file path

    Returns
    -------
    output : tuple of ndarray and dict
        Returns a 2-tuple of the array of stacked projections and input_meta.
    """

    # Retrieve initialized norm matrix

    reset_norm = __plugin_state__['reset_norm']

    # Raise error if input is not a numpy array

    if not hasattr(input_data, 'dtype'):
        raise ValueError('invalid resetproj preprocess input array')

    input_data[:] = reset_norm

    return (input_data, input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
