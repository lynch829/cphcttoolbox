#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# npykatsevich - numpy katsevich reconstruction
# Copyright (C) 2011-2013  The CT-Toolbox Project lead by Brian Vinter
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

"""Spiral cone beam CT using the Katsevich algorithm in numpy"""

import sys
import traceback

from cphct.io import create_path_dir
from cphct.log import logging, allowed_log_levels, setup_log, \
    default_level, log_scan_geometry
from cphct.misc import timelog

# These are basic numpy functions exposed through npy to use same numpy

from cphct.npycore import pi, cos, arctan, ceil
from cphct.npycore.io import save_auto
from cphct.npycore.io import npy_free_all, get_npy_data, \
    get_npy_total_size
from cphct.npycore.misc import slide_forward
from cphct.npycore.utils import log_checksum
from cphct.plugins import load_plugins, execute_plugin
from cphct.cone.katsevich.conf import default_katsevich_npy_conf, \
    default_katsevich_npy_opts, parse_setup, ParseError
from cphct.cone.katsevich.npy.io import fill_katsevich_npy_conf
from cphct.cone.katsevich.npycore.kernels import init_recon, \
    filter_chunk, backproject_chunk

app_names = ['katsevich', 'npykatsevich']


def reconstruct_volume(conf, npy_plugins):
    """Reconstruct 3D volume from the recorded 2D projections

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    npy_plugins : dict
        A dictionary of numpy plugins to use

    Returns
    -------
    output : dict
        Dictionary of configuration options with updated timelog
    """

    # Create save path dirs and empty binary save files

    if conf['save_filtered_projs_data_path']:
        create_path_dir(conf['save_filtered_projs_data_path'])
        open(conf['save_filtered_projs_data_path'], 'wb', 0).close()

    # Check Tam-Danielsson window coverage and warn if not covered

    if conf['detector_shape'] == 'flat':
        win_min_rows = ceil(1 + conf['progress_per_turn'] / (pi
                            * conf['scan_radius'] * conf['scan_diameter'
                            ] * conf['detector_pixel_height'])
                            * (conf['detector_half_width'] ** 2
                            + conf['scan_diameter'] ** 2) * (pi / 2
                            + arctan(conf['detector_half_width']
                            / conf['scan_diameter'])))
        win_min_cols = ceil(1 + 2 * conf['detector_half_width']
                            / conf['detector_pixel_width'])
    elif conf['detector_shape'] == 'curved':
        win_min_rows = ceil(1 + conf['scan_diameter']
                            * conf['progress_per_turn'] * (pi / 2
                            + conf['half_fan_angle']) / (pi
                            * conf['scan_radius']
                            * conf['detector_pixel_height']
                            * cos(conf['half_fan_angle'])))
        win_min_cols = ceil(1 + 2 * conf['half_fan_angle']
                            / conf['detector_pixel_width'])

    if win_min_rows > conf['detector_rows'] or win_min_cols \
        > conf['detector_columns']:
        logging.warning('Tam-Danielsson window not covered!')
        logging.warning('min rows %d vs %d and min cols %d vs %d'
                        % (win_min_rows, conf['detector_rows'],
                        win_min_cols, conf['detector_columns']))

    # Build list of chunk indices

    chunk_limits = []
    z_voxels = conf['z_voxels']
    chunk_size = conf['chunk_size']
    for i in xrange(conf['chunk_count']):
        (first_z, last_z) = (i * chunk_size, min(z_voxels, (i + 1)
                             * chunk_size) - 1)
        first_proj = i * conf['chunk_projs_offset']

        # last_proj is index of the last projection used for back projection.

        last_proj = first_proj + conf['chunk_projs'] - 1
        chunk_limits.append((first_z, last_z, first_proj, last_proj))

    if not chunk_limits:
        logging.error('no chunks enabled!')
        sys.exit(1)

    # Filter runs on a single rotation of projs at a time, which is independent
    # of the reconstruction chunk size so we use filter_* buffers for the
    # filtering and input_chunk and output_chunk for reconstruction.

    # Matrices for storage in host memory

    filter_in = get_npy_data(conf, 'filter_in')
    filter_diff = get_npy_data(conf, 'filter_diff')
    filter_rebin = get_npy_data(conf, 'filter_rebin')
    hilbert_ideal = get_npy_data(conf, 'hilbert_ideal')
    filter_conv = get_npy_data(conf, 'filter_conv')
    filter_out = get_npy_data(conf, 'filter_out')
    input_buffer = get_npy_data(conf, 'input_buffer')
    input_chunk = get_npy_data(conf, 'input_chunk')
    proj_row_mins = get_npy_data(conf, 'proj_row_mins')
    proj_row_maxs = get_npy_data(conf, 'proj_row_maxs')
    output_chunk = get_npy_data(conf, 'output_chunk')

    logging.debug('using about %db of host memory'
                  % get_npy_total_size(conf))

    (last_filtered, first_filtered) = (-1, 0)

    # Keep output files open for chunked writing

    if conf['save_filtered_projs_data_path']:
        filtered_projs_data_fd = \
            open(conf['save_filtered_projs_data_path'], 'wb', 0)

    for chunk in xrange(conf['chunk_count']):
        if not chunk in conf['chunks_enabled']:
            continue

        # Tell plugins which chunk we are processing

        conf['app_state']['chunk']['idx'] = chunk

        # Chunk projection indexes are those of the filtered projections

        (first_z, last_z, first_proj, last_proj) = chunk_limits[chunk]
        end_proj = last_proj + 1
        (extended_first, extended_last) = (first_proj, last_proj)

        # Extend input_chunk projs range to a multiplum of filter_out_projs.
        # Filtering to prepare all those projections takes place but only the
        # ones from first_proj to last_proj are used in this particular back
        # projection chunk

        if first_proj % conf['filter_out_projs'] != 0:
            extended_first = conf['filter_out_projs'] * int(first_proj
                    / conf['filter_out_projs'])
        if last_proj % conf['filter_out_projs'] != 0:
            extended_end = conf['filter_out_projs'] * int(1 + last_proj
                    / conf['filter_out_projs'])
            extended_last = extended_end - 1
        else:
            extended_end = extended_last + 1

        # Filter needs extra projection(s) to feed back projection

        input_first = extended_first
        input_last = extended_last + conf['extra_filter_projs']
        logging.debug('reconstruct z %d - %d from projs %d - %d (%d - %d)'
                       % (
            first_z,
            last_z,
            first_proj,
            last_proj,
            input_first,
            input_last,
            ))

        buffer_switch = extended_first - first_filtered
        if buffer_switch > 0:

            # Shift already filtered projs for reuse

            logging.debug('shift projs %d' % buffer_switch)
            slide_forward(input_buffer, buffer_switch)

        # reuse already filtered projections and just continue from there

        next_first = max(last_filtered + 1, extended_first)

        timelog.set(conf, 'verbose', 'filter')
        for j in xrange(next_first, extended_last,
                        conf['filter_out_projs']):

            # Act on projections from in_first to in_last (inclusive) creating
            # filtered projection from out_first to out_last (inclusive).
            # Automatically limit to indices to fit total_projs.

            (in_first, in_end) = (j, j + conf['filter_in_projs'])
            in_end = min(in_end, conf['total_projs'])
            in_last = in_end - 1
            (out_first, out_end) = (j, j + conf['filter_out_projs'])
            out_end = min(out_end, conf['total_projs'])
            out_last = out_end - 1

            logging.debug('filter from %d to %d (%d - %d)' % (in_first,
                          in_last, extended_first, extended_last))

            logging.debug('loading raw projs from %d to %d'
                          % (in_first, in_last))

            # Load exactly the projections for use in filtering

            # Tell plugins which projection range we are processing

            conf['app_state']['projs']['first'] = in_first
            conf['app_state']['projs']['last'] = in_last

            # View for plugins

            in_size = in_end - in_first
            projs_data = filter_in[:in_size]
            projs_meta = []

            hook = 'npy_load_input'
            logging.info('Loading with %s numpy plugin(s)'
                         % ', '.join([plug[0] for plug in
                         npy_plugins[hook]]))
            req_npy_plugins = npy_plugins.get(hook, [])
            for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                logging.debug('Loading chunk with %s plugin' % name)
                try:
                    timelog.set(conf, 'default', 'proj_load')

                    # Always pass projs_data, projs_meta and conf as first args

                    execute_plugin(hook, name, plugin_mod, [projs_data,
                                   projs_meta, conf] + args, kwargs)

                    timelog.log(conf, 'default', 'proj_load')
                except Exception:
                    logging.error('Load plugin %s failed:\n%s' % (name,
                                  traceback.format_exc()))
                    sys.exit(1)

            if conf['checksum']:
                chunk_view = projs_data.ravel()
                log_checksum('raw projs part', chunk_view,
                             chunk_view.size)

            filter_meta = projs_meta

            # Preprocess current chunk of projections with numpy plugins

            hook = 'npy_preprocess_input'
            logging.info('Preprocessing with %s numpy plugin(s)'
                         % ', '.join([plug[0] for plug in
                         npy_plugins[hook]]))
            req_npy_plugins = npy_plugins.get(hook, [])
            for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                logging.debug('Preprocessing chunk with %s numpy plugin'
                               % name)
                try:
                    timelog.set(conf, 'default', 'npy_preprocess')

                    # Always pass filter_in, filter_meta and conf as first
                    # args

                    execute_plugin(hook, name, plugin_mod, [filter_in,
                                   filter_meta, conf] + args, kwargs)

                    timelog.log(conf, 'default', 'npy_preprocess')
                except Exception:
                    logging.error('Preprocess numpy plugin %s failed:\n%s'
                                   % (name, traceback.format_exc()))

                    sys.exit(1)

            if conf['checksum']:
                chunk_view = filter_in.ravel()
                log_checksum('projs part', chunk_view, chunk_view.size)

            # Skip filtering step if projections are already on filtered form
            # Accept empty projs_meta to allow dummy runs without input

            if projs_meta and projs_meta[-1]['filtered']:
                out_size = out_end - out_first
                filter_out[:] = filter_in[:out_size]
            else:
                (filter_diff[:], filter_rebin[:]) = (0, 0)
                (filter_conv[:], filter_out[:]) = (0, 0)
                timelog.log(conf, 'verbose', 'core_filter',
                            start_time=0.0, end_time=filter_chunk(
                    chunk,
                    in_first,
                    in_last,
                    filter_in,
                    filter_diff,
                    filter_rebin,
                    hilbert_ideal,            
                    filter_conv,
                    filter_out,
                    conf,
                    ))
                if conf['checksum']:
                    log_checksum('filter diff', filter_diff,
                                 filter_diff.size)
                    log_checksum('filter rebin', filter_rebin,
                                 filter_rebin.size)
                    log_checksum('filter conv', filter_conv,
                                 filter_conv.size)
                    log_checksum('filter out', filter_out,
                                 filter_out.size)

            logging.debug('saving filtered projs %d - %d' % (out_first,
                          out_last))

            # Include last projection

            rel_in_first = out_first - extended_first
            rel_in_end = out_end - extended_first
            rel_out_end = out_end - out_first
            input_buffer[rel_in_first:rel_in_end] = \
                filter_out[:rel_out_end]

            if conf['save_filtered_projs_data_path']:
                timelog.set(conf, 'verbose', 'proj_save')
                save_auto(filtered_projs_data_fd, filter_out)
                timelog.log(conf, 'verbose', 'proj_save')

            if conf['checksum']:
                log_checksum('filter out', filter_out, filter_out.size)

        # Update index for last filtered projection

        (last_filtered, first_filtered) = (extended_last,
                extended_first)
        rel_in_first = first_proj - extended_first
        rel_in_end = end_proj - extended_first

        # TODO: just use a view instead of copy

        input_chunk[:] = input_buffer[rel_in_first:rel_in_end]

        filter_time = timelog.log(conf, 'verbose', 'filter')
        msg = 'finished filtering projs'
        if conf['timelog'] == 'verbose':
            msg = '%s in %ss' % (msg, filter_time)

        logging.debug(msg)

        if conf['checksum']:
            log_checksum('filtered projs', input_chunk,
                         input_chunk.size)

        timelog.set(conf, 'verbose', 'backproject')

        if conf['checksum']:
            chunk_view = input_chunk.ravel()
            log_checksum('filtered part', chunk_view, chunk_view.size)

        logging.debug('run reconstruction of chunk %d:%d with projs %d:%d'
                       % (first_z, last_z, first_proj, last_proj))
        output_chunk[:] = 0
        recon_meta = []

        timelog.set(conf, 'verbose', 'core_backproject')

        backproject_chunk(chunk, first_proj, last_proj, first_z, last_z,
                          input_chunk, proj_row_mins, proj_row_maxs,
                          output_chunk, conf)

        log_time = timelog.log(conf, 'verbose', 'core_backproject')

        msg = 'Reconstructed chunk: %s:%s' % (first_z, last_z)

        if conf['timelog'] == 'verbose':
            msg = '%s in %.4f seconds' % (msg, log_time)

        logging.info(msg)

        conf['app_state']['chunk']['layout'] = ('x', 'y', 'z')

        if conf['checksum']:
            log_checksum('result', output_chunk, output_chunk.size)

        # Scale out values to compensate for summed contribution from multiple
        # projections (this is a delayed part of the back projection formula)

        output_chunk /= conf['projs_per_turn']

        # We reconstruct in x, y, z but the raw output format is z, y, x

        output_zyx = output_chunk.swapaxes(0, 2)

        # View for plugins

        recon_chunk = output_zyx[:]
        conf['app_state']['chunk']['layout'] = ('z', 'y', 'x')

        timelog.log(conf, 'verbose', 'backproject')

        # Postprocess current chunk of results with numpy plugins

        hook = 'npy_postprocess_output'
        logging.info('Postprocessing with %s numpy plugin(s)'
                     % ', '.join([plug[0] for plug in
                     npy_plugins[hook]]))
        req_npy_plugins = npy_plugins.get(hook, [])
        for (name, plugin_mod, args, kwargs) in req_npy_plugins:
            logging.debug('Postprocessing chunk with %s numpy plugin'
                          % name)
            try:
                timelog.set(conf, 'default', 'npy_postprocess')

                # Always pass recon_chunk, recon_meta and conf as first args

                execute_plugin(hook, name, plugin_mod, [recon_chunk,
                               recon_meta, conf] + args, kwargs)

                timelog.log(conf, 'default', 'npy_postprocess')
            except Exception:
                logging.error('Postprocess numpy plugin %s failed:\n%s'
                              % (name, traceback.format_exc()))
                sys.exit(1)

        if conf['checksum']:
            chunk_view = output_zyx.ravel()
            log_checksum('postprocessed part', chunk_view,
                         chunk_view.size)

        hook = 'npy_save_output'
        logging.info('Saving with %s numpy plugin(s)'
                     % ', '.join([plug[0] for plug in
                     npy_plugins[hook]]))
        req_npy_plugins = npy_plugins.get(hook, [])

        for (name, plugin_mod, args, kwargs) in req_npy_plugins:
            logging.debug('Saving chunk with %s plugin' % name)
            try:
                timelog.set(conf, 'default', 'recon_save')

                # Always pass recon_chunk, recon_meta and conf as first args

                execute_plugin(hook, name, plugin_mod, [recon_chunk,
                               recon_meta, conf] + args, kwargs)

                timelog.log(conf, 'default', 'recon_save')
            except Exception:
                logging.error('Save plugin %s failed:\n%s' % (name,
                              traceback.format_exc()))
                sys.exit(1)

    # Close file descriptors used for chunked writing

    if conf['save_filtered_projs_data_path']:
        filtered_projs_data_fd.close()

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
        'core_filter',
        'filter',
        'core_backproject',
        'backproject',
        'proj_save',
        'npy_plugin_exit',
        'npy_memory_clean',
        ]

    timelog.init(conf, default, verbose)

    timelog.set(conf, 'default', 'complete')

    # Complete configuration initialization

    timelog.set(conf, 'verbose', 'conf_init')
    fill_katsevich_npy_conf(conf)
    fdt = conf['data_type']
    timelog.log(conf, 'verbose', 'conf_init')

    if conf['detector_row_offset'] != 0.0:
        logging.warning('Katsevich detector_row_offset is experimental!'
                        )

    # Initialize Katsevich kernel data structures

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

    logging.info('Starting %(detector_shape)s Katsevich reconstruction'
                 % conf)
    log_scan_geometry(conf, opts)
    logging.debug('Full conf: %s' % conf)

    # Start reconstruction

    timelog.set(conf, 'default', 'recon_volume')
    reconstruct_volume(conf, npy_plugins)
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

        logging.info('Kernel times:')
        logging.info('  filtering:             %.4fs'
                     % timelog.get(conf, 'verbose', 'filter'))
        logging.info('  backprojection:        %.4fs'
                     % timelog.get(conf, 'verbose', 'backproject'))
        logging.info('  Core:')
        logging.info('    filtering:           %.4fs'
                     % timelog.get(conf, 'verbose', 'core_filter'))
        logging.info('    backprojection:      %.4fs'
                     % timelog.get(conf, 'verbose', 'core_backproject'))

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
    print 'Run Katsevich reconstruction'


if __name__ == '__main__':
    npy_cfg = default_katsevich_npy_conf()
    npy_opts = default_katsevich_npy_opts()
    try:
        npy_cfg = parse_setup(sys.argv, app_names, npy_opts, npy_cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(1)
    main(npy_cfg, npy_opts)
