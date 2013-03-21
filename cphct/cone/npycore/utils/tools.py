#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# tools - cone beam back end functions shared by tools
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

"""Cph CT Toolbox cone beam back end functions shared by tools.
We try to generalize helper functions so that they can be used
inside all tools scripts.
"""

import os
import sys
import time

from cphct.cone.conf import str_value, int_value
from cphct.cone.npy.io import fill_cone_npy_conf
from cphct.conf import allowed_engines, default_engine, bool_value
from cphct.io import path_expander, collapse_path, create_path_dir
from cphct.log import logging, setup_log, allowed_log_levels, \
    default_level
from cphct.npycore import zeros, allowed_data_types
from cphct.npycore.io import get_npy_data, get_npy_total_size, \
    npy_alloc, npy_free_all, save_auto
from cphct.plugins import load_plugins, execute_plugin


def default_tool_opts():
    """Default options for tools

    Returns
    -------
    output : dict
        Returns a dictionary of default tool options helper dictionaries.
    """

    opts = {
        'save_projs_image_path': {
            'long': 'save-projs-image-path',
            'short': None,
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': 'projection%.4d.raw',
            'description': 'Path pattern for output projections, "%d" is ' \
                + 'replaced with index',
            },
        'save_projs_data_path': {
            'long': 'save-projs-data-path',
            'short': None,
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': 'projections.bin',
            'description': 'Path for binary dump of output projections',
            },
        'save_projs_scene_path': {
            'long': 'save-projs-scene-path',
            'short': None,
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': 'scene.csv',
            'description': 'Path for output projections scene description',
            },
        'use_relative_scene_paths': {
            'long': 'use-relative-scene-paths',
            'short': None,
            'args': bool,
            'handler': bool_value,
            'default': False,
            'description': 'Use relative projection file paths in scene file',
            },
        'chunk_projs': {
            'long': 'chunk-projs',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 1,
            'description': 'Number of projections to process at a time.',
            },
        }

    return opts


def default_init_tool(conf):
    """Default init tool helpers

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : dict
        Returns a dictionary of general tool helper variables.
    """

    state = {}

    # Default to preserved layout.
    # Open scene file for subsequent use if it is enabled.

    state['output_projs'] = conf['chunk_projs']
    state['output_rows'] = conf['detector_rows']
    state['output_columns'] = conf['detector_columns']

    if conf['save_projs_scene_path']:
        create_path_dir(conf['save_projs_scene_path'])
        state['scene_file'] = open(conf['save_projs_scene_path'], 'w')
    return state


def default_core_tool(projs, conf, tool_state):
    """Default core tool helpers used if no additional processing is needed

    Parameters
    ----------
    projs : ndarray
        Array with chunk of projections to process.
    conf : dict
        A dictionary of configuration options.
    tool_state : dict
        A dictionary of tool helper variables.
    """

    return projs


def default_exit_tool(conf, tool_state):
    """Default tool helpers clean up

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    tool_state : dict
        A dictionary of tool helper variables.
    """

    if conf['save_projs_scene_path']:
        tool_state['scene_file'].close()
    tool_state.clear()


def __tool_helper(
    app_names,
    conf,
    init_tool,
    core_tool,
    exit_tool,
    inline,
    ):
    """Helper to generate processed projections using either the general method
    of creating new arrays/views of the loaded projections or the inline
    version where the loaded projections are modified directly. For some tools
    it is possible to enable the more efficient inline version that modifies
    the loaded projections directly, but that is only the case if the
    projection shape remains unchanged.

    Parameters
    ----------
    app_names: list of str
        A list of names that the application accepts
    conf : dict
        A dictionary of configuration options.
    init_tool : function
        A function that initializes the particular tool based on the conf
        dictionary after basic setup. Must return a tool state dictionary with
        any variables for use in subsequent tool functions.
    core_tool : function
        A function that takes the loaded chunk of projections, the conf and
        tool state dictionaries and returns a processed version of the chunk.
    exit_tool : function
        A function that cleans up after the particular tool based on the conf
        and tool state dictionaries.
    inline: boolean
        Flag to tell the helper to apply the tool action directly on the loaded
        projections rather than creating a new possibly reshaped copy or view.

    Returns
    -------
    output : ndarray
        Array with chunk of processed projections.
    """

    before_complete = time.time()
    if conf['log_level']:
        conf['log_level'] = allowed_log_levels[conf['log_level']]
    else:
        conf['log_level'] = default_level
    setup_log(conf['log_path'], conf['log_level'], conf['log_format'])
    timelog = {'preprocess': 0.0, 'proj_load': 0.0, 'proj_save': 0.0}
    fdt = conf['data_type']
    odt = conf['output_data_type']

    # Complete configuration initialization

    fill_cone_npy_conf(conf)

    if conf['total_turns'] < 0:
        conf['total_turns'] = 1
        logging.info('defaulting to one rotation')

    total_projs = conf['projs_per_turn'] * conf['total_turns']
    if total_projs % conf['chunk_projs'] != 0:
        logging.error('total number of projections (%d) must be a multiplum of chunk_projs (%d)!'
                       % (total_projs, conf['chunk_projs']))
        return

    # Tool specific init

    tool_state = init_tool(conf)

    # Create save path dirs and empty binary save file

    if conf['save_projs_data_path']:
        create_path_dir(conf['save_projs_data_path'])
        save_projs_fd = open(conf['save_projs_data_path'], 'wb')

    # Init projection meta data list

    projs_meta = []

    # Allocate memory for projection data.
    # Please note that input is forced to internal data type during processing.

    npy_alloc(conf, 'projs_data', zeros((conf['chunk_projs'],
              conf['detector_rows'], conf['detector_columns']),
              dtype=fdt))
    projs_data = get_npy_data(conf, 'projs_data')

    # Allocate memory for output data if not using inline modification.
    # Force same output format in inline mode.

    if not inline:
        output_projs = tool_state.get('output_projs', conf['chunk_projs'
                ])
        output_rows = tool_state.get('output_rows', conf['detector_rows'
                ])
        output_columns = tool_state.get('output_columns',
                conf['detector_columns'])
        npy_alloc(conf, 'out_data', zeros((output_projs, output_rows,
                  output_columns), dtype=odt))
        out_data = get_npy_data(conf, 'out_data')
    else:
        tool_state['output_rows'] = conf['chunk_projs']
        tool_state['output_rows'] = conf['detector_rows']
        tool_state['output_columns'] = conf['detector_columns']

    # Load numpy plugins

    (npy_plugins, errors) = load_plugins(app_names, 'npy', conf)
    for (key, val) in errors.items():
        for (plugin_name, load_err) in val:
            logging.error('loading %s %s numpy plugin failed : %s'
                          % (key, plugin_name, load_err))
            sys.exit(1)

    # Prepare configured numpy plugins

    hook = 'npy_plugin_init'
    logging.info('Initializing %s numpy plugin(s)' % ', '.join([plug[0]
                 for plug in npy_plugins[hook]]))
    req_npy_plugins = npy_plugins.get(hook, [])
    for (name, plugin_mod, args, kwargs) in req_npy_plugins:
        logging.debug('Initialize %s numpy plugin' % name)
        try:

            # Always pass conf as first arg

            execute_plugin(hook, name, plugin_mod, [conf] + args,
                           kwargs)
        except Exception, exc:
            logging.error('Init numpy plugin %s failed:\n%s' % (name,
                          exc))
            sys.exit(1)

    before_total = time.time()

    logging.info('----  Preprocessing projections  ----')
    logging.debug('Full conf: %s' % conf)
    proj_count = 0
    for step in xrange(conf['total_turns']):
        logging.info('Preprocessing step: %s/%s' % (step + 1,
                     conf['total_turns']))

        conf['app_state']['scanner_step'] = step

        base_index = step * conf['projs_per_turn']

        for proj in xrange(conf['projs_per_turn']):
            if proj % conf['chunk_projs'] != 0:
                continue

            # Load next projection chunk

            proj_index = base_index + proj
            end_index = min(proj_index + conf['chunk_projs'],
                            base_index + conf['projs_per_turn'])
            last_index = end_index - 1

            # Tell plugins which projection range we are processing

            conf['app_state']['projs']['first'] = proj_index
            conf['app_state']['projs']['last'] = last_index

            before_load = time.time()

            hook = 'npy_load_input'
            req_npy_plugins = npy_plugins.get(hook, [])
            for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                logging.info('Loading chunk with %s plugin' % name)
                logging.debug('%s args, kwargs: %s %s %s' % (name,
                              plugin_mod, args, kwargs))
                try:

                    # Always pass projs_data, projs_meta and conf as
                    # first args

                    execute_plugin(hook, name, plugin_mod, [projs_data,
                                   projs_meta, conf] + args, kwargs)
                except Exception, exc:
                    logging.error('Load plugin %s failed:\n%s' % (name,
                                  exc))
                    import traceback
                    logging.error(traceback.format_exc())
                    sys.exit(1)

            after_load = time.time()
            timelog['proj_load'] += after_load - before_load

            # Preprocess current chunk of projections
            # with configured plugins

            logging.debug('process %s chunk of projs with state %s'
                          % (projs_data.shape, tool_state))
            before_preprocess = time.time()

            hook = 'npy_preprocess_input'
            req_npy_plugins = npy_plugins.get(hook, [])
            for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                logging.info('Preprocessing chunk with %s plugin'
                             % name)
                logging.debug('%s args, kwargs: %s %s %s' % (name,
                              plugin_mod, args, kwargs))
                try:

                    # Always pass projs_data, proj_meta and conf as
                    # first args

                    execute_plugin(hook, name, plugin_mod, [projs_data,
                                   projs_meta, conf] + args, kwargs)
                except Exception, exc:
                    logging.error('Preprocess plugin %s failed:\n%s'
                                  % (name, exc))
                    sys.exit(1)

            # Finally apply any specific processing

            if inline:
                core_tool(projs_data, conf, tool_state)
                out_data = projs_data
            else:
                out_data[:] = core_tool(projs_data, conf, tool_state)

            after_preprocess = time.time()
            timelog['preprocess'] += after_preprocess \
                - before_preprocess

            for chunk_offset in xrange(out_data.shape[0]):
                proj_global = proj_index + chunk_offset
                logging.debug('Write preprocessed %s proj %d with sum %f'
                               % (projs_data.shape, proj_global,
                              projs_data.sum()))
                if conf['save_projs_image_path']:
                    save_path = conf['save_projs_image_path'] \
                        % proj_global
                    save_auto(save_path, out_data[chunk_offset])
                    logging.debug('Wrote preprocessed proj %d in %s'
                                  % (proj_global, save_path))
                if conf['save_projs_data_path']:
                    save_auto(save_projs_fd, out_data[chunk_offset])
                    logging.debug('Wrote preprocessed proj %d in %s'
                                  % (proj_global, conf['save_projs_data_path']))
                if conf['save_projs_scene_path']:
                    proj_angle = projs_meta[chunk_offset]['angle']
                    proj_path = save_path
                    if conf['use_relative_scene_paths']:
                        proj_path = \
                            collapse_path(conf['working_directory'],
                                proj_path)
                    scene_entry = ', '.join([proj_path, proj_angle])
                    tool_state['scene_file'].write('%s\n' % scene_entry)
                    logging.debug('Wrote scene entry: %s' % scene_entry)
                proj_count += 1

    if conf['save_projs_image_path']:
        logging.info('Wrote %d preprocessed projs in %s' % (proj_count,
                     os.path.dirname(conf['save_projs_image_path'])))
    if conf['save_projs_data_path']:
        save_projs_fd.close()
        logging.info('Wrote %d preprocessed projs in %s' % (proj_count,
                     conf['save_projs_data_path']))
    if conf['save_projs_scene_path']:
        logging.info('Wrote %d preprocessed scene entries in %s'
                     % (proj_count, conf['save_projs_scene_path']))

    # TODO: add transfer_missing_metadata support?

    logging.info('Host memory used for preprocessing %db'
                 % get_npy_total_size(conf))
    logging.info('IO times:')
    logging.info('  load projections:     %.4fs' % timelog['proj_load'])
    logging.info('  save projections:     %.4fs' % timelog['proj_save'])
    logging.info('Plugin times:')
    logging.info('  preprocess:     %.4fs' % timelog['preprocess'])

    # Clean up after numpy plugins

    hook = 'npy_plugin_exit'
    logging.info('Cleaning up after %s numpy plugin(s)'
                 % ', '.join([plug[0] for plug in npy_plugins[hook]]))
    req_npy_plugins = npy_plugins.get(hook, [])
    for (name, plugin_mod, args, kwargs) in req_npy_plugins:
        logging.debug('Clean up %s numpy plugin' % name)
        try:

            # Always pass conf as first arg

            execute_plugin(hook, name, plugin_mod, [conf] + args,
                           kwargs)
        except Exception, exc:
            logging.error('Exit numpy plugin %s failed:\n%s' % (name,
                          exc))
            sys.exit(1)

    # Clean up numpy memory

    npy_free_all(conf)

    # Tool-specific clean up

    exit_tool(conf, tool_state)

    after_total = after_complete = time.time()
    total_time = after_total - before_total
    complete_time = after_complete - before_complete

    logging.info('Total run time %.3fs' % total_time)
    logging.info('Complete time used %.3fs' % complete_time)
    logging.shutdown()


def general_tool_helper(
    app_names,
    conf,
    init_tool,
    core_tool,
    exit_tool,
    ):
    """Generate processed projections using the general method of creating
    new arrays/views of the loaded projections. For some tools it is possible
    to use the more efficient inline_tool_helper that modifies the loaded
    projections directly.

    Parameters
    ----------
    app_names: list of str
        A list of names that the application accepts
    conf : dict
        A dictionary of configuration options.
    init_tool : function
        A function that initializes the particular tool based on the conf
        dictionary after basic setup. Must return a tool state dictionary with
        any variables for use in subsequent tool functions.
    core_tool : function
        A function that takes the loaded chunk of projections, the conf and
        tool state dictionaries and returns a processed version of the chunk.
    exit_tool : function
        A function that cleans up after the particular tool based on the conf
        and tool state dictionaries.

    Returns
    -------
    output : ndarray
        Array with chunk of processed projections.
    """

    return __tool_helper(
        app_names,
        conf,
        init_tool,
        core_tool,
        exit_tool,
        inline=False,
        )


def inline_tool_helper(
    app_names,
    conf,
    init_tool,
    core_tool,
    exit_tool,
    ):
    """Generate processed projections using the optimized method of editing
    the loaded projections inline. For the tools not modifying the projection
    shape it is possible to use this more efficient function instead of the
    general_tool_helper that relies on copies.

    Parameters
    ----------
    app_names: list of str
        A list of names that the application accepts
    conf : dict
        A dictionary of configuration options.
    init_tool : function
        A function that initializes the particular tool based on the conf
        dictionary after basic setup. Must return a tool state dictionary with
        any variables for use in subsequent tool functions.
    core_tool : function
        A function that takes the loaded chunk of projections, the conf and
        tool state dictionaries and processes the chunk inline.
    exit_tool : function
        A function that cleans up after the particular tool based on the conf
        and tool state dictionaries.

    Returns
    -------
    output : ndarray
        Array with chunk of processed projections.
    """

    return __tool_helper(
        app_names,
        conf,
        init_tool,
        core_tool,
        exit_tool,
        inline=True,
        )


def engine_app_aliases(app_names, conf):
    """Return a list of engine specific aliases for the names in th elist of
    app_names based on the engine value in conf or the default engine if not
    set.

    app_names: list of str
        A list of names that the application accepts
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : list of str
        Array with engine-specific aliases for names in app_names.
    """

    if not conf['engine']:
        engine_prefix = allowed_engines[default_engine]
    else:
        engine_prefix = allowed_engines[conf['engine']]
    return ['%s%s' % (engine_prefix, name) for name in app_names]


