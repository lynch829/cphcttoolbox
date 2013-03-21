#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# dummy - dummy cuda plugin to illustrate plugin functionality
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

"""Simple dummy cuda plugin that may be used as a template for developing your
own plugins for input and output processing.

General plugin function structure is:
 * check that function input is compatible with the pre/post processor
   - raise exception or print warning otherwise
 * do actual processing directly in data array if possible
 * return processed data array (same if possible)

Please note that you may pass additional args for the processing with
the 'PLUGIN#POSARG1#..#POSARGN#ARGNAME1=NAMEDARG1#ARGNAMEN=NAMEDARGN' format.
Then they will show up as *string* values in the args tuple and the kwargs
dictionary respectively. Any conversion from strings to other types is left to
the plugin.
"""

from cphct.log import logging

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(conf, *args, **kwargs):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    *args : positional arguments
        Any positional arguments.
    **kwargs : keyword arguments
        Any keyword arguments.
    """

    __plugin_state__['name'] = __name__
    logging.debug('in dummy cuda plugin init')


def plugin_exit(conf, *args, **kwargs):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    *args : positional arguments
        Any positional arguments.
    **kwargs : keyword arguments
        Any keyword arguments.
    """

    __plugin_state__.clear()
    logging.debug('in dummy cuda plugin exit')


def load_input(
    input_data,
    input_meta,
    conf,
    *args,
    **kwargs
    ):
    """Dummy cuda load function to use as a custom plugin sample

    Parameters
    ----------
    input_data : gpuarray
        Array to load projections into.
    input_meta : list of dict
        List of meta data to fill with dictionaries matching each projection.
    conf : dict
        A dictionary of configuration options.
    *args : positional arguments
        Any positional arguments.
    **kwargs : keyword arguments
        Any keyword arguments.

    Returns
    -------
    output : tuple of array and list
        Returns a tuple of the same data array and meta list.
    """

    logging.debug('in dummy cuda plugin loading')
    logging.debug('''state: %s
args: %s
kwargs: %s''' % (__plugin_state__,
            args, kwargs))
    return (input_data, input_meta)


def preprocess_input(
    input_data,
    input_meta,
    conf,
    *args,
    **kwargs
    ):
    """Dummy cuda preprocessing function to use as a custom plugin sample

    Parameters
    ----------
    input_data : gpuarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    *args : positional arguments
        Any positional arguments.
    **kwargs : keyword arguments
        Any keyword arguments.

    Returns
    -------
    output : tuple of gpuarray and list
        Returns a tuple of the same data array and meta list.
    """

    logging.debug(':in dummy cuda plugin preprocessing')
    logging.debug('''state: %s
args: %s
kwargs: %s''' % (__plugin_state__,
            args, kwargs))
    return (input_data, input_meta)


def postprocess_output(
    output_data,
    output_meta,
    conf,
    *args,
    **kwargs
    ):
    """Dummy cuda postprocessing function to use as a custom plugin sample

    Parameters
    ----------
    output_data : gpuarray
        array to process.
    output_meta : list of dict
        List of meta data dictionaries matching output_data.
    conf : dict
        A dictionary of configuration options.
    *args : positional arguments
        Any positional arguments.
    **kwargs : keyword arguments
        Any keyword arguments.

    Returns
    -------
    output : tuple of gpuarray and dict
        Returns a tuple of the same data array and meta list.
    """

    logging.debug('in dummy cuda plugin postprocessing')
    logging.debug('''state: %s
args: %s
kwargs: %s''' % (__plugin_state__,
            args, kwargs))
    return (output_data, output_meta)


def save_output(
    output_data,
    output_meta,
    conf,
    *args,
    **kwargs
    ):
    """Dummy cuda save function to use as a custom plugin sample

    Parameters
    ----------
    output_data : gpuarray
        Array of reconstructed volume voxels to save from.
    output_meta : list of dict
        List of meta data for volume.
    conf : dict
        A dictionary of configuration options.
    *args : positional arguments
        Any positional arguments.
    **kwargs : keyword arguments
        Any keyword arguments.

    Returns
    -------
    output : tuple of array and list
        Returns a tuple of the same data array and meta list.
    """

    logging.debug('in dummy cuda plugin saving')
    logging.debug('''state: %s
args: %s
kwargs: %s''' % (__plugin_state__,
            args, kwargs))
    return (output_data, output_meta)


if __name__ == '__main__':
    print 'no unit tests!'
