#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# loadscene - plugin to load input data from a scene file
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

# TODO: move load_projs_chunk to utils?

"""Loads input data for reconstruction from a scene file where each line is
semi-colon separated list of a projection file path and meta data. Allows
override of the individual projection files with a single binary stacked
projection file.
"""

from cphct.io import expand_path
from cphct.log import logging
from cphct.npycore.io import load_projs_chunk

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(
    conf,
    scene_path,
    prefiltered=False,
    projs_path=None,
    ):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Check input paths.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    scene_path : str
        Where to load scene from.
    prefiltered : str formatted bool
        Optional marker to point out that loaded projections are already on
        prefiltered form. The projection meta data will include this
        information so that the prefiltering step is skipped later.
    projs_path : str or None
        Where to optionally load projections for scene override from.

    Raises
    ------
    IOError
        If provided scene_path is not a valid file or if projs_path is set but
        does not point to a file.
    """

    # Parse prefiltered *string* value to a boolean

    if str(prefiltered).lower() in ('1', 't', 'true', 'y', 'yes'):
        __plugin_state__['prefiltered_projs'] = True
    elif str(prefiltered).lower() in (
        '',
        '0',
        'f',
        'false',
        'n',
        'no',
        ):
        __plugin_state__['prefiltered_projs'] = False
    else:
        logging.warning('unexpected "prefiltered" argument "%s"!')
        __plugin_state__['prefiltered_projs'] = False

    # Expand paths and open files for efficient chunked reading

    abs_scene_path = expand_path(conf['working_directory'], scene_path)
    __plugin_state__['abs_scene_path'] = abs_scene_path
    __plugin_state__['scene_fd'] = open(abs_scene_path, 'r')
    if projs_path:
        abs_projs_path = expand_path(conf['working_directory'],
                projs_path)
        __plugin_state__['abs_projs_path'] = abs_projs_path
        __plugin_state__['projs_fd'] = open(abs_projs_path, 'rb')
    else:
        __plugin_state__['abs_projs_path'] = None
        __plugin_state__['projs_fd'] = None


def plugin_exit(
    conf,
    scene_path,
    prefiltered=False,
    projs_path=None,
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
    scene_path : str
        Where to load scene from.
    prefiltered : bool
        Optional marker to point out that loaded projections are already on
        prefiltered form. The projection meta data will include this
        information so that the prefiltering step is skipped later.
    projs_path : str or None
        Where to optionally load projections for scene override from.
    """

    # Close files again

    __plugin_state__['scene_fd'].close()
    if projs_path:
        __plugin_state__['projs_fd'].close()
    __plugin_state__.clear()


def load_input(
    input_data,
    input_meta,
    conf,
    scene_path,
    prefiltered=False,
    projs_path=None,
    ):
    """Load projections with meta data from a scene file and optionally
    override the actual projections with ones from a binary dump.

    Parameters
    ----------
    input_data : ndarray
        Array to load projections into.
    input_meta : list of dict
        List of meta data to fill with dictionaries matching each projection.
    conf : dict
        A dictionary of configuration options.
    scene_path : str
        Where to load scene from.
    prefiltered : bool
        Optional marker to point out that loaded projections are already on
        prefiltered form. The projection meta data will include this
        information so that the prefiltering step is skipped later.
    projs_path : str or None
        Where to optionally load projections for scene override from.
        
    Raises
    -------
    ValueError :
        If scene or projections can't be loaded.
    """

    # Use in-place update to projs_list and projs_data

    (load_meta, load_data) = load_projs_chunk(
        conf['app_state']['projs']['first'],
        conf['app_state']['projs']['last'],
        conf['working_directory'],
        __plugin_state__['abs_scene_path'],
        __plugin_state__['abs_projs_path'],
        conf['detector_rows'],
        conf['detector_columns'],
        conf['input_data_type'],
        conf['data_type'],
        __plugin_state__['prefiltered_projs'],
        )

    input_meta[:] = load_meta
    input_data[:load_data.shape[0]] = load_data
    
    return (input_data, input_meta)


if __name__ == '__main__':
    print 'no unit tests!'
