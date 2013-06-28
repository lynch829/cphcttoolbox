#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# npycenterfdk - numpy center slice fdk reconstruction
# Copyright (C) 2011-2012  The CT-Toolbox Project lead by Brian Vinter
#
# This file is part of CT-Toolbox.
#
# CT-Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# CT-Toolbox is distributed in the hope that it will be useful,
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

"""Circular fan beam CT using the center slice FDK algorithm in numpy"""

import sys
import traceback

from cphct.io import create_path_dir
from cphct.log import logging, allowed_log_levels, setup_log, \
    default_level, log_scan_geometry
from cphct.misc import timelog

# These are basic numpy functions exposed through npy to use same numpy

from cphct.npycore.io import get_npy_data, get_npy_total_size, \
    npy_free_all
from cphct.npycore.utils import log_checksum
from cphct.plugins import load_plugins, execute_plugin
from cphct.fan.centerfdk.conf import default_centerfdk_npy_conf, \
    default_centerfdk_npy_opts, parse_setup, ParseError
from cphct.fan.centerfdk.npy.io import fill_centerfdk_npy_conf
from cphct.fan.centerfdk.npycore.kernels import init_recon, \
    reconstruct_proj

app_names = ['centerfdk', 'npycenterfdk']


def reconstruct_volume(conf, fdt, npy_plugins):
    """Reconstruct 3D volume from the recorded 2D projections

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    fdt : dtype
        Float data type (internal precision)
    npy_plugins : dict
        A dictionary of numpy plugins to use

    Returns
    -------
    output : dict
        Dictionary of configuration options with updated timelog
    """

    if conf['checksum']:
        proj_weight_matrix = get_npy_data(conf, 'proj_weight_matrix')
        proj_weight_view = proj_weight_matrix.ravel()
        log_checksum('proj_weight', proj_weight_view,
                     proj_weight_view.size)

        proj_filter_matrix = get_npy_data(conf, 'proj_filter_matrix')
        proj_filter_view = proj_filter_matrix.ravel()
        log_checksum('proj_filter', proj_filter_view,
                     proj_filter_view.size)

    # Create save path dirs and empty binary save files

    if conf['save_filtered_projs_data_path']:
        create_path_dir(conf['save_filtered_projs_data_path'])
        open(conf['save_filtered_projs_data_path'], 'wb', 0).close()

    # Init projection and recon meta data lists

    (projs_meta, recon_meta) = ([], [])

    # Get pre-allocated projection and recon chunk data

    projs_data = get_npy_data(conf, 'projs_data')
    recon_chunk = get_npy_data(conf, 'recon_chunk')

    # Loop over the amount of steps in the scene

    for step in xrange(conf['total_turns']):
        logging.info('Reconstructing step: %s/%s' % (step,
                     conf['total_turns']))

        conf['app_state']['scanner_step'] = step

        # NOTE: We re-read every projection if volume is split into chunks,
        # This might be improved

        # Loop over the amount of volume chunks in the z direction

        for chunk in xrange(conf['chunk_count']):
            if not chunk in conf['chunks_enabled']:
                continue

            # Tell plugins which chunk we are processing

            conf['app_state']['chunk']['idx'] = chunk

            z_voxels_start = chunk * conf['chunk_size']
            z_voxels_end = z_voxels_start + conf['chunk_size']

            logging.info('Reconstructing chunk: %d, z-voxel: %s -> %s'
                         % (chunk, z_voxels_start, z_voxels_end))

            # Reset recon chunk data

            timelog.set(conf, 'verbose', 'reset_recon_chunk')
            recon_chunk[:] = fdt(0)
            timelog.log(conf, 'verbose', 'reset_recon_chunk')

            z_voxels_array = get_npy_data(conf, 'z_voxels_coordinates'
                    )[z_voxels_start:z_voxels_end]

            # Loop over the projections in each chunk for each step

            for proj in xrange(conf['projs_per_turn']):

                # Load next projection chunk

                proj_index = step * conf['projs_per_turn'] + proj

                # Tell plugins which projection range we are processing

                conf['app_state']['projs']['first'] = proj_index
                conf['app_state']['projs']['last'] = proj_index

                hook = 'npy_load_input'
                logging.info('Loading chunk with %s plugin(s)'
                             % ', '.join([plug[0] for plug in
                             npy_plugins[hook]]))

                req_npy_plugins = npy_plugins.get(hook, [])
                for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                    logging.debug('Loading chunk with %s plugin' % name)

                    try:
                        timelog.set(conf, 'default', 'proj_load')

                        # Always pass projs_data, projs_meta and conf as
                        # first args

                        execute_plugin(hook, name, plugin_mod,
                                [projs_data, projs_meta, conf] + args,
                                kwargs)
                        timelog.log(conf, 'default', 'proj_load')
                    except Exception:
                        logging.error('Load plugin %s failed:\n%s'
                                % (name, traceback.format_exc()))
                        sys.exit(1)

                if conf['checksum']:
                    chunk_view = projs_data.ravel()
                    log_checksum('Raw projs chunk', chunk_view,
                                 chunk_view.size)

                # Preprocess current chunk of projections
                # with configured numpy plugins

                hook = 'npy_preprocess_input'
                req_npy_plugins = npy_plugins.get(hook, [])

                logging.info('Preprocessing with %s numpy plugin(s)'
                             % ', '.join([plug[0] for plug in
                             npy_plugins[hook]]))

                for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                    logging.debug('Preprocessing chunk with %s numpy plugin'
                                   % name)
                    try:
                        timelog.set(conf, 'default', 'npy_preprocess')

                        # Always pass projs_data, proj_meta and conf as
                        # first args

                        execute_plugin(hook, name, plugin_mod,
                                [projs_data, projs_meta, conf] + args,
                                kwargs)

                        timelog.log(conf, 'default', 'npy_preprocess')
                    except Exception:
                        logging.error('Preprocess numpy plugin %s failed:\n%s'
                                 % (name, traceback.format_exc()))
                        sys.exit(1)

                if conf['checksum']:
                    chunk_view = projs_data.ravel()
                    log_checksum('Preprocessed projs chunk',
                                 chunk_view, chunk_view.size)

                for load_index in xrange(len(projs_meta)):
                    conf['app_state']['backproject']['proj_idx'] = \
                        proj_index + load_index

                    proj_meta = projs_meta[load_index]

                    # Reconstruct the loaded projection

                    timelog.set(conf, 'verbose', 'proj_recon')

                    reconstruct_proj(conf, proj_meta, z_voxels_array,
                            fdt)

                    log_time = timelog.log(conf, 'verbose', 'proj_recon'
                            )

                    msg = 'Reconstructed projection: %s, angle: %s' \
                        % (conf['app_state']['backproject']['proj_idx'
                           ], proj_meta['angle'])

                    if conf['timelog'] == 'verbose':
                        msg = '%s in %.4f seconds' % (msg, log_time)

                    logging.info(msg)

                    if conf['checksum']:
                        recon_chunk_view = recon_chunk.ravel()
                        log_checksum('Recon chunk', recon_chunk_view,
                                recon_chunk_view.size)

            # Postprocess current chunk of results with configured numpy plugins

            hook = 'npy_postprocess_output'
            req_npy_plugins = npy_plugins.get(hook, [])
            logging.info('Postprocessing with %s numpy plugin(s)'
                         % ', '.join([plug[0] for plug in
                         npy_plugins[hook]]))

            for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                logging.debug('Postprocessing chunk with %s numpy plugin'
                               % name)
                try:
                    timelog.set(conf, 'default', 'npy_postprocess')

                    # Always pass recon_chunk, recon_meta and conf as first
                    # args

                    execute_plugin(hook, name, plugin_mod,
                                   [recon_chunk, recon_meta, conf]
                                   + args, kwargs)

                    timelog.log(conf, 'default', 'npy_postprocess')
                except Exception:
                    logging.error('Postprocess numpy plugin %s failed:\n%s'
                                   % (name, traceback.format_exc()))
                    sys.exit(1)

            if conf['checksum']:
                recon_chunk_view = recon_chunk.ravel()
                log_checksum('Final recon chunk', recon_chunk_view,
                             recon_chunk_view.size)

            hook = 'npy_save_output'
            req_npy_plugins = npy_plugins.get(hook, [])
            logging.info('Saving chunk with %s plugin(s)'
                         % ', '.join([plug[0] for plug in
                         npy_plugins[hook]]))

            for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                logging.debug('Saving chunk with %s plugin' % name)

                try:
                    timelog.set(conf, 'default', 'recon_save')

                    # Always pass recon_chunk, recon_meta and conf as first
                    # args

                    execute_plugin(hook, name, plugin_mod,
                                   [recon_chunk, recon_meta, conf]
                                   + args, kwargs)

                    timelog.log(conf, 'default', 'recon_save')
                except Exception:
                    logging.error('Save plugin %s failed:\n%s' % (name,
                                  traceback.format_exc()))
                    sys.exit(1)

    return conf


def main(conf, opts):
    """Run entire reconstruction using settings from conf dictionary

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    opts : dict
        A dictionary of application options.
    """

    if conf['log_level']:
        conf['log_level'] = allowed_log_levels[conf['log_level']]
    else:
        conf['log_level'] = default_level
    setup_log(conf['log_path'], conf['log_level'], conf['log_format'])

    # Initialize timelog

    default = [
        'complete',
        'recon_volume',
        'proj_load',
        'recon_save',
        'npy_preprocess',
        'npy_postprocess',
        ]
    verbose = [
        'conf_init',
        'npy_plugin_init',
        'reset_recon_chunk',
        'proj_weight',
        'proj_filter',
        'proj_save',
        'transform_matrix',
        'volume_weight',
        'backproject',
        'proj_recon',
        'npy_plugin_exit',
        'npy_memory_clean',
        ]

    timelog.init(conf, default, verbose)

    timelog.set(conf, 'default', 'complete')

    # Complete configuration initialization

    timelog.set(conf, 'verbose', 'conf_init')
    fill_centerfdk_npy_conf(conf)
    fdt = conf['data_type']
    timelog.log(conf, 'verbose', 'conf_init')

    # Initialize FDK kernel data structures

    init_recon(conf, fdt)

    # Load numpy plugins

    timelog.set(conf, 'verbose', 'npy_plugin_init')
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
        except Exception:
            logging.error('Init numpy plugin %s failed:\n%s' % (name,
                          traceback.format_exc()))
            sys.exit(1)
    timelog.log(conf, 'verbose', 'npy_plugin_init')

    logging.info('Starting %(detector_shape)s FDK reconstruction'
                 % conf)
    log_scan_geometry(conf, opts)
    logging.debug('Full conf: %s' % conf)

    # Start reconstruction

    timelog.set(conf, 'default', 'recon_volume')
    reconstruct_volume(conf, fdt, npy_plugins)
    timelog.log(conf, 'default', 'recon_volume')

    # Get memory usage for logging before cleanup

    total_npy_memory_usage = get_npy_total_size(conf)

    # Clean up after numpy plugins

    timelog.set(conf, 'verbose', 'npy_plugin_exit')
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
        except Exception:
            logging.error('Exit numpy plugin %s failed:\n%s' % (name,
                          traceback.format_exc()))
            sys.exit(1)
    timelog.log(conf, 'verbose', 'npy_plugin_exit')

    # Clean up numpy memory

    timelog.set(conf, 'verbose', 'npy_memory_clean')
    npy_free_all(conf)
    timelog.log(conf, 'verbose', 'npy_memory_clean')

    timelog.log(conf, 'default', 'complete')

    logging.info('--------------- Recon Timings -----------------')
    logging.info('Ran entire reconstruction in: %.4fs'
                 % timelog.get(conf, 'default', 'recon_volume'))
    logging.info('(proj. avg. %.4fs) (chunk avg. %.4fs)'
                 % (timelog.get(conf, 'default', 'recon_volume')
                 / conf['total_projs'], timelog.get(conf, 'default',
                 'recon_volume') / conf['chunk_count']))

    logging.info('Memory usage:')
    logging.info('  main: %dMB (%dB)' % (total_npy_memory_usage / 1024
                 ** 2, total_npy_memory_usage))

    if conf['timelog'] == 'verbose':
        logging.info('Init times:')
        logging.info('  conf:                  %.4fs'
                     % timelog.get(conf, 'verbose', 'conf_init'))
        logging.info('  numpy plugins:         %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_plugin_init'))

    logging.info('IO times:')
    logging.info('  load projections:      %.4fs' % timelog.get(conf,
                 'default', 'proj_load'))
    logging.info('  save recon chunks:     %.4fs' % timelog.get(conf,
                 'default', 'recon_save'))

    if conf['timelog'] == 'verbose':
        logging.info('  save projections:      %.4fs'
                     % timelog.get(conf, 'verbose', 'proj_save'))
        logging.info('  reset recon chunk:     %.4fs'
                     % timelog.get(conf, 'verbose', 'reset_recon_chunk'
                     ))
        logging.info('Kernel times:')
        logging.info('  projection weight:     %.4fs'
                     % timelog.get(conf, 'verbose', 'proj_weight'))
        logging.info('  projection filter:     %.4fs'
                     % timelog.get(conf, 'verbose', 'proj_filter'))
        logging.info('  transform matrix:      %.4fs'
                     % timelog.get(conf, 'verbose', 'transform_matrix'))
        logging.info('  volume weight:         %.4fs'
                     % timelog.get(conf, 'verbose', 'volume_weight'))
        logging.info('  backproject:           %.4fs'
                     % timelog.get(conf, 'verbose', 'backproject'))

    logging.info('Plugin times:')
    logging.info('  numpy preprocess:      %.4fs' % timelog.get(conf,
                 'default', 'npy_preprocess'))
    logging.info('  numpy postprocess:     %.4fs' % timelog.get(conf,
                 'default', 'npy_postprocess'))

    if conf['timelog'] == 'verbose':
        logging.info('Cleanup times:')
        logging.info('  numpy memory:          %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_memory_clean'))
        logging.info('  numpy plugins:         %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_plugin_exit'))

    logging.info('Complete time used %.3fs' % timelog.get(conf,
                 'default', 'complete'))
    logging.shutdown()


def usage():
    """Usage help"""

    print 'Usage: %s' % sys.argv[0]
    print 'Run center slice FDK reconstruction'


if __name__ == '__main__':
    npy_cfg = default_centerfdk_npy_conf()
    npy_opts = default_centerfdk_npy_opts()
    try:
        npy_cfg = parse_setup(sys.argv, app_names, npy_opts, npy_cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(1)
    main(npy_cfg, npy_opts)
