#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# cukatsevich - OpenCL katsevich reconstruction
# Copyright (C) 2011-2014  The CT-Toolbox Project lead by Brian Vinter
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

"""Spiral cone beam CT using the Katsevich algorithm in OpenCL"""

import sys
import traceback

from cphct.cl.core import gpu_init_mod, gpu_init_ctx, gpu_device_count, \
    gpu_kernels_auto_init, gpu_exit, log_gpu_specs, gpu_save_kernels, \
    gpu_alloc_from_array, gpu_mem_info
from cphct.cl.io import cl_free_all, get_cl_data, get_cl_total_size
from cphct.io import create_path_dir
from cphct.log import logging, allowed_log_levels, setup_log, \
    default_level, log_scan_geometry
from cphct.misc import timelog

# These are basic NumPy functions exposed through npy to use same NumPy

from cphct.npycore import pi, cos, arctan, ceil, allowed_data_types
from cphct.npycore.io import save_auto
from cphct.npycore.io import npy_free_all, get_npy_data, \
    get_npy_total_size
from cphct.npycore.misc import slide_forward
from cphct.npycore.utils import log_checksum
from cphct.plugins import load_plugins, execute_plugin
from cphct.cone.katsevich.conf import default_katsevich_cl_conf, \
    default_katsevich_cl_opts, parse_setup, ParseError
from cphct.cone.katsevich.cl.io import fill_katsevich_cl_conf
from cphct.cone.katsevich.cl.kernels import init_recon, filter_chunk, \
    backproject_chunk, rt_const

app_names = ['katsevich', 'clkatsevich']


def reconstruct_volume(conf, npy_plugins, cl_plugins):
    """Reconstruct 3D volume from the recorded 2D projections

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    npy_plugins : dict
        A dictionary of NumPy plugins to use
    cl_plugins : dict
        A dictionary of CUDA plugins to use

    Returns
    -------
    output : dict
        Dictionary of configuration options with updated timelog
    """

    int32 = allowed_data_types['int32']

    # Create save path dirs and empty binary save files

    if conf['save_filtered_projs_data_path']:
        create_path_dir(conf['save_filtered_projs_data_path'])
        open(conf['save_filtered_projs_data_path'], 'wb', 0).close()

    gpu_queue = conf['gpu']['queue']
    checksum_array = conf['app_state']['gpu']['prepared_kernels'
            ]['checksum_array']

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
        gpu_exit(conf)
        conf['app_state']['exit_code'] = 1
        return conf

    # Filter runs on a single rotation of projs at a time, which is independent
    # of the reconstruction chunk size so we use filter_* buffers for the
    # filtering and input_projs, input_chunk and recon_chunk for
    # reconstruction.

    chunk_items = conf['chunk_projs'] * conf['detector_rows'] \
        * conf['detector_columns']

    if conf['checksum']:
        (check_first, check_last) = (0, chunk_items)

    # Matrices for storage in host memory

    filter_in = get_npy_data(conf, 'filter_in')
    filter_diff = get_npy_data(conf, 'filter_diff')
    filter_rebin = get_npy_data(conf, 'filter_rebin')
    filter_conv = get_npy_data(conf, 'filter_conv')
    filter_out = get_npy_data(conf, 'filter_out')
    input_buffer = get_npy_data(conf, 'input_buffer')
    input_projs = get_npy_data(conf, 'input_projs')
    input_chunk = get_npy_data(conf, 'input_chunk')
    recon_chunk = get_npy_data(conf, 'recon_chunk')
    if conf['checksum']:
        check_dest = get_npy_data(conf, 'check_dest')

    # Matrices for storage in device memory

    gpu_filter_in = get_cl_data(conf, 'filter_in')
    gpu_filter_diff = get_cl_data(conf, 'filter_diff')
    gpu_filter_rebin = get_cl_data(conf, 'filter_rebin')
    gpu_proj_filter_array = get_cl_data(conf, 'proj_filter_array')
    gpu_filter_conv = get_cl_data(conf, 'filter_conv')
    gpu_filter_out = get_cl_data(conf, 'filter_out')
    gpu_input_chunk = get_cl_data(conf, 'input_chunk')
    gpu_proj_row_mins = get_cl_data(conf, 'proj_row_mins')
    gpu_proj_row_maxs = get_cl_data(conf, 'proj_row_maxs')
    gpu_recon_chunk = get_cl_data(conf, 'recon_chunk')

    if conf['checksum']:
        gpu_check_dest = get_cl_data(conf, 'check_dest')

    logging.debug('using about %db of host memory'
                  % get_npy_total_size(conf))

    logging.debug('using about %db of device memory'
                  % get_cl_total_size(conf))

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

        # Extend input_projs range to a multiplum of filter_out_projs.
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

        # TODO: this is horribly inefficient for big projs (mem explosion)
        # ... can we either make it a memmove or maybe interface a fixed
        # buffer as a circular buffer with a helper function to retrieve
        # projection X when needed. On-demand filtering with caching would be
        # even better.
        # numpy.roll(input_buffer, -buffer_switch, axis=0) may be better?

        buffer_switch = extended_first - first_filtered
        if buffer_switch > 0:

            # Shift already filtered projs for reuse

            logging.debug('shift projs %d' % buffer_switch)
            slide_forward(input_buffer, buffer_switch)

        # reuse already filtered projections and just continue from there

        next_first = max(last_filtered + 1, extended_first)

        timelog.set(conf, 'verbose', 'filter', barrier=True)
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
            logging.info('Loading with %s NumPy plugin(s)'
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
                    gpu_exit(conf)
                    conf['app_state']['exit_code'] = 1
                    return conf

            if conf['checksum']:
                chunk_view = projs_data.ravel()
                log_checksum('raw projs part', chunk_view,
                             chunk_view.size)

            filter_meta = projs_meta

            # Preprocess current chunk of projections with NumPy plugins

            hook = 'npy_preprocess_input'
            logging.info('Preprocessing with %s NumPy plugin(s)'
                         % ', '.join([plug[0] for plug in
                         npy_plugins[hook]]))
            req_npy_plugins = npy_plugins.get(hook, [])
            for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                logging.debug('Preprocessing chunk with %s NumPy plugin'
                               % name)
                try:
                    timelog.set(conf, 'default', 'npy_preprocess')

                    # Always pass filter_in, filter_meta and conf as first
                    # args

                    execute_plugin(hook, name, plugin_mod, [filter_in,
                                   filter_meta, conf] + args, kwargs)

                    timelog.log(conf, 'default', 'npy_preprocess')
                except Exception:
                    logging.error('Preprocess NumPy plugin %s failed:\n%s'
                                   % (name, traceback.format_exc()))
                    gpu_exit(conf)
                    conf['app_state']['exit_code'] = 1
                    return conf

            if conf['checksum']:
                chunk_view = filter_in.ravel()
                log_checksum('projs part', chunk_view, chunk_view.size)

            # Skip filtering step if projections are already on filtered form
            # Accept empty projs_meta to allow dummy runs without input

            if conf['proj_filter'] == 'skip' or projs_meta \
                and projs_meta[-1]['filtered']:
                out_size = out_end - out_first
                filter_out[:] = filter_in[:out_size]
            else:
                (filter_diff[:], filter_rebin[:]) = (0, 0)
                (filter_conv[:], filter_out[:]) = (0, 0)

                # Reset GPU data
                # NOTE: Due to memory leak in pyopencl.array.Array.fill
                #       we call .finish to cleanup in each iteration

                timelog.set(conf, 'verbose', 'memset_gpu')
                gpu_filter_diff.finish()
                gpu_filter_diff.fill(0, queue=gpu_queue)
                gpu_filter_rebin.finish()
                gpu_filter_rebin.fill(0, queue=gpu_queue)
                gpu_filter_conv.finish()
                gpu_filter_conv.fill(0, queue=gpu_queue)
                gpu_filter_out.finish()
                gpu_filter_out.fill(0, queue=gpu_queue)
                timelog.log(conf, 'verbose', 'memset_gpu', barrier=True)
                logging.debug('copy %d (%db) projs from host to dev'
                              % (in_end - in_first, filter_in.nbytes))
                timelog.set(conf, 'verbose', 'host_to_gpu')
                gpu_filter_in.set(filter_in, queue=gpu_queue)
                timelog.log(conf, 'verbose', 'host_to_gpu',
                            barrier=True)

                # Preprocess current chunk of projections with CUDA plugins

                # Expose input as gpuarray to plugins

                hook = 'cl_preprocess_input'
                logging.info('Preprocessing with %s OpenCL plugin(s)'
                             % ', '.join([plug[0] for plug in
                             cl_plugins[hook]]))
                req_cl_plugins = cl_plugins.get(hook, [])
                for (name, plugin_mod, args, kwargs) in req_cl_plugins:
                    logging.debug('Preprocessing chunk with %s OpenCL plugin'
                                   % name)
                    try:
                        timelog.set(conf, 'default', 'cl_preprocess',
                                    barrier=True)

                        # Always pass filter_in and conf as first args

                        execute_plugin(hook, name, plugin_mod,
                                [gpu_filter_in, filter_meta, conf]
                                + args, kwargs)

                        timelog.log(conf, 'default', 'cl_preprocess',
                                    barrier=True)
                    except Exception:
                        logging.error('Preprocess OpenCL plugin %s failed:\n%s'
                                 % (name, traceback.format_exc()))
                        gpu_exit(conf)
                        conf['app_state']['exit_code'] = 1
                        return conf

                timelog.set(conf, 'verbose', 'core_filter',
                            barrier=True)

                filter_chunk(
                    gpu_queue,
                    in_first,
                    in_last,
                    gpu_filter_in,
                    gpu_filter_diff,
                    gpu_filter_rebin,
                    gpu_proj_filter_array,
                    gpu_filter_conv,
                    gpu_filter_out,
                    conf,
                    )
                timelog.log(conf, 'verbose', 'core_filter',
                            barrier=True)

                logging.debug('copy %d (%db) filtered projs from dev to host'
                               % (len(filter_out), filter_out.nbytes))

                # TODO: copy only to gpu_input_chunk if in gpu_projs_only mode
                # Technically we could filter directly into gpu_input_chunk with
                # an offset but handling the projs extension may be to much of a
                # hassle compared to the device-to-device copy cost.
                # i.e. somtehing like
                # gpu_input_chunk_offset = gpu_module.intp(gpu_input_chunk+offset)
                # gpu_module.memcpy_dtod(gpu_input_chunk_offset, gpu_filter_out)

                timelog.set(conf, 'verbose', 'gpu_to_host')
                gpu_filter_out.get(queue=gpu_queue, ary=filter_out)
                timelog.log(conf, 'verbose', 'gpu_to_host',
                            barrier=True)

                if conf['checksum']:
                    timelog.set(conf, 'verbose', 'gpu_to_host')
                    gpu_filter_diff.get(queue=gpu_queue,
                            ary=filter_diff)
                    timelog.log(conf, 'verbose', 'gpu_to_host',
                                barrier=True)

                    log_checksum('gpu filter diff', filter_diff,
                                 filter_diff.size)

                    timelog.set(conf, 'verbose', 'gpu_to_host')
                    gpu_filter_rebin.get(queue=gpu_queue,
                            ary=filter_rebin)
                    timelog.log(conf, 'verbose', 'gpu_to_host',
                                barrier=True)

                    log_checksum('gpu filter rebin', filter_rebin,
                                 filter_rebin.size)

                    timelog.set(conf, 'verbose', 'gpu_to_host')
                    gpu_filter_conv.get(queue=gpu_queue,
                            ary=filter_conv)
                    timelog.log(conf, 'verbose', 'gpu_to_host',
                                barrier=True)

                    log_checksum('gpu filter conv', filter_conv,
                                 filter_conv.size)
                    log_checksum('gpu filter out', filter_out,
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

        input_projs[:] = input_buffer[rel_in_first:rel_in_end]

        filter_time = timelog.log(conf, 'verbose', 'filter',
                                  barrier=True)
        msg = 'finished filtering projs'
        if conf['timelog'] == 'verbose':
            msg = '%s in %ss' % (msg, filter_time)

        logging.debug(msg)

        if conf['checksum']:
            log_checksum('filtered projs', input_projs,
                         input_projs.size)

        timelog.set(conf, 'verbose', 'backproject', barrier=True)

        recon_meta = []
        timelog.set(conf, 'verbose', 'memset_gpu', barrier=True)
        gpu_recon_chunk.finish()
        gpu_recon_chunk.fill(0)
        timelog.log(conf, 'verbose', 'memset_gpu', barrier=True)
        log_time = 0.0

        # Reconstruct using chunks of projections to scale to big projections

        for j in xrange(first_proj, last_proj + 1,
                        conf['backproject_in_projs']):

            # Act on a sub-range of projections from first_proj to last_proj
            # (inclusive) updating all z slices in chunk range with the
            # contribution from those projections.

            (in_first, in_end) = (j, j + conf['backproject_in_projs'])
            in_end = min(in_end, conf['total_projs'])
            in_last = in_end - 1
            rel_first = in_first - first_proj
            rel_end = in_end - first_proj

            logging.debug('backproject projections %d to %d (%d - %d)'
                          % (in_first, in_last, first_proj, last_proj))

            input_chunk[:] = input_projs[rel_first:rel_end]

            logging.debug('copy %d (%db) projs from no. %d forward to device'
                           % (len(input_chunk), input_chunk.nbytes,
                          in_first))
            logging.debug('copy %s to %s' % (input_chunk.size,
                          gpu_input_chunk))

            # TODO: only copy back to gpu if not in gpu_projs_only mode

            timelog.set(conf, 'verbose', 'host_to_gpu')
            gpu_input_chunk.set(input_chunk, queue=gpu_queue)
            timelog.log(conf, 'verbose', 'host_to_gpu', barrier=True)

            if conf['checksum']:
                gpu_check_dest.finish()
                gpu_check_dest.fill(0.0)
                grid = block = (1, 1, 1)
                checksum_array(
                    gpu_queue,
                    grid,
                    block,
                    gpu_alloc_from_array(gpu_check_dest),
                    gpu_alloc_from_array(gpu_input_chunk),
                    int32(in_first),
                    int32(in_last),
                    )

                gpu_check_dest.get(queue=gpu_queue, ary=check_dest)
                logging.debug('test checksum_array %d - %d projs is %f'
                              % (in_first, in_last, check_dest[0]))

            logging.debug('run reconstruction of chunk %d:%d with projs %d:%d'
                           % (first_z, last_z, in_first, in_last))

            timelog.set(conf, 'verbose', 'core_backproject',
                        barrier=True)

            backproject_chunk(
                gpu_queue,
                chunk,
                in_first,
                in_last,
                first_z,
                last_z,
                gpu_input_chunk,
                gpu_proj_row_mins,
                gpu_proj_row_maxs,
                gpu_recon_chunk,
                conf,
                )

            log_time += timelog.log(conf, 'verbose', 'core_backproject'
                                    , barrier=True)

        msg = 'Reconstructed chunk: %s:%s' % (first_z, last_z)

        if conf['timelog'] == 'verbose':
            msg = '%s in %.4f seconds' % (msg, log_time)

        logging.info(msg)

        conf['app_state']['chunk']['layout'] = ('x', 'y', 'z')

        # Postprocess current chunk of results with CUDA plugins

        hook = 'cl_postprocess_output'
        logging.info('Postprocessing with %s OpenCL plugin(s)'
                     % ', '.join([plug[0] for plug in
                     cl_plugins[hook]]))
        req_cl_plugins = cl_plugins.get(hook, [])
        for (name, plugin_mod, args, kwargs) in req_cl_plugins:
            logging.debug('Postprocessing chunk with %s OpenCL plugin'
                          % name)
            try:
                timelog.set(conf, 'default', 'cl_postprocess',
                            barrier=True)

                # Always pass gpu_recon_chunk and conf as first args

                execute_plugin(hook, name, plugin_mod,
                               [gpu_recon_chunk, recon_meta, conf]
                               + args, kwargs)

                timelog.log(conf, 'default', 'cl_postprocess',
                            barrier=True)
            except Exception:
                logging.error('Postprocess OpenCL plugin %s failed:\n%s'
                               % (name, traceback.format_exc()))
                gpu_exit(conf)
                conf['app_state']['exit_code'] = 1
                return conf

        timelog.set(conf, 'verbose', 'gpu_to_host')
        gpu_recon_chunk.get(queue=gpu_queue, ary=recon_chunk)
        timelog.log(conf, 'verbose', 'gpu_to_host', barrier=True)
        if conf['checksum']:
            gpu_check_dest.finish()
            gpu_check_dest.fill(0.0)
            grid = block = (1, 1, 1)
            checksum_array(
                gpu_queue,
                grid,
                block,
                gpu_alloc_from_array(gpu_check_dest),
                gpu_alloc_from_array(gpu_recon_chunk),
                int32(in_first),
                int32(in_last),
                )

            gpu_check_dest.get(queue=gpu_queue, ary=check_dest)

            logging.debug('test checksum_array %d - %d result is %f'
                          % (check_first, check_last, check_dest[0]))

            log_checksum('result', recon_chunk, recon_chunk.size)

        # We reconstruct in x, y, z but the raw output format is z, y, x

        conf['app_state']['chunk']['layout'] = ('z', 'y', 'x')

        timelog.log(conf, 'verbose', 'backproject', barrier=True)

        # Postprocess current chunk of results with NumPy plugins

        hook = 'npy_postprocess_output'
        logging.info('Postprocessing with %s NumPy plugin(s)'
                     % ', '.join([plug[0] for plug in
                     npy_plugins[hook]]))
        req_npy_plugins = npy_plugins.get(hook, [])
        for (name, plugin_mod, args, kwargs) in req_npy_plugins:
            logging.debug('Postprocessing chunk with %s NumPy plugin'
                          % name)
            try:
                timelog.set(conf, 'default', 'npy_postprocess')

                # Always pass recon_chunk, recon_meta and conf as first args

                execute_plugin(hook, name, plugin_mod, [recon_chunk,
                               recon_meta, conf] + args, kwargs)

                timelog.log(conf, 'default', 'npy_postprocess')
            except Exception:
                logging.error('Postprocess NumPy plugin %s failed:\n%s'
                              % (name, traceback.format_exc()))
                gpu_exit(conf)
                conf['app_state']['exit_code'] = 1
                return conf

        if conf['checksum']:
            chunk_view = recon_chunk.ravel()
            log_checksum('postprocessed part', chunk_view,
                         chunk_view.size)

        hook = 'npy_save_output'
        logging.info('Saving with %s NumPy plugin(s)'
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
                gpu_exit(conf)
                conf['app_state']['exit_code'] = 1
                return conf

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

    Returns
    -------
    output : int
        An integer exit code for the run, 0 means success
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
        'cl_preprocess',
        'cl_postprocess',
        ]

    if conf['log_level'] == logging.DEBUG:
        default.extend(['diff_chunk', 'fwd_rebin_chunk',
                       'convolve_chunk', 'rev_rebin_chunk',
                       'backproject_chunk'])

    verbose = [
        'conf_init',
        'npy_plugin_init',
        'cl_plugin_init',
        'gpu_init',
        'gpu_kernel',
        'memset_gpu',
        'host_to_gpu',
        'gpu_to_host',
        'core_filter',
        'filter',
        'core_backproject',
        'backproject',
        'proj_save',
        'cl_plugin_exit',
        'npy_plugin_exit',
        'cl_memory_clean',
        'npy_memory_clean',
        'gpu_exit',
        ]

    timelog.init(conf, default, verbose)

    timelog.set(conf, 'default', 'complete')

    # Complete configuration initialization

    timelog.set(conf, 'verbose', 'conf_init')
    fill_katsevich_cl_conf(conf)

    fdt = conf['data_type']
    timelog.log(conf, 'verbose', 'conf_init')

    if conf['detector_row_offset'] != 0.0:
        logging.warning('Katsevich detector_row_offset is experimental!'
                        )

    # Check that z-range fits recording without overscan half-rotations

    z_range = conf['z_max'] - conf['z_min']
    recon_progress = conf['progress_per_turn'] * conf['core_turns']
    if abs(recon_progress - z_range) > 0.01 * conf['progress_per_turn']:
        msg = 'Katsevich z-range must fit total_turns'
        msg = '%s and progress_per_turn! (%f vs %f)' % (msg, z_range,
                recon_progress)
        logging.warning(msg)

    # Init GPU access

    logging.info('Init GPU %d' % conf['gpu_device_index'])
    timelog.set(conf, 'verbose', 'gpu_init')
    gpu_init_mod(conf)
    gpu_count = gpu_device_count(conf)
    if gpu_count < 1:
        logging.error('No GPUs available!')
        conf['app_state']['exit_code'] = 1
        return conf['app_state']['exit_code']
    if conf['gpu_device_index'] > gpu_count - 1:
        logging.error('GPU device index (%d) must match GPU IDs! (%s)'
                      % (conf['gpu_device_index'], ', '.join(['%s' % i
                      for i in range(gpu_count)])))
        conf['app_state']['exit_code'] = 1
        return conf['app_state']['exit_code']
    if conf['gpu_device_index'] < 0:

        # Just default to the first one for now - 'best' would be better

        conf['gpu_device_index'] = 0

    gpu_init_ctx(conf)
    timelog.log(conf, 'verbose', 'gpu_init')

    # Prepare GPU inlining of variables that are constant at kernel runtime

    log_gpu_specs(conf)
    (gpu_free, gpu_total) = gpu_mem_info(conf)
    logging.info('GPU memory: %dMB (%dB) free of %dMB (%dB) total'
                 % (gpu_free / 1024 ** 2, gpu_free, gpu_total / 1024
                 ** 2, gpu_total))

    timelog.set(conf, 'verbose', 'gpu_kernel')
    gpu_kernels_auto_init(conf, rt_const)
    gpu_save_kernels(conf)
    timelog.log(conf, 'verbose', 'gpu_kernel')

    if not conf['cl_kernels']:
        logging.error('no valid gpu compute kernels found!')
        gpu_exit(conf)
        conf['app_state']['exit_code'] = 1
        return conf['app_state']['exit_code']

    logging.debug('filter out projs %d' % conf['filter_out_projs'])
    logging.debug('backproj in projs %d' % conf['backproject_in_projs'])

    # Initialize Katsevich kernel data structures

    init_recon(conf, fdt)

    # Load NumPy plugins

    timelog.set(conf, 'verbose', 'npy_plugin_init')
    (npy_plugins, errors) = load_plugins(app_names, 'npy', conf)
    for (key, val) in errors.items():
        for (plugin_name, load_err) in val:
            logging.error('loading %s %s NumPy plugin failed : %s'
                          % (key, plugin_name, load_err))
            gpu_exit(conf)
            conf['app_state']['exit_code'] = 1
            return conf['app_state']['exit_code']

    # Prepare configured NumPy plugins

    hook = 'npy_plugin_init'
    logging.info('Initializing %s NumPy plugin(s)' % ', '.join([plug[0]
                 for plug in npy_plugins[hook]]))
    req_npy_plugins = npy_plugins.get(hook, [])
    for (name, plugin_mod, args, kwargs) in req_npy_plugins:
        logging.debug('Initialize %s NumPy plugin' % name)
        try:

            # Always pass conf as first arg

            execute_plugin(hook, name, plugin_mod, [conf] + args,
                           kwargs)
        except Exception:
            logging.error('Init NumPy plugin %s failed:\n%s' % (name,
                          traceback.format_exc()))
            gpu_exit(conf)
            conf['app_state']['exit_code'] = 1
            return conf['app_state']['exit_code']
    timelog.log(conf, 'verbose', 'npy_plugin_init')

    # Load OpenCL plugins

    timelog.set(conf, 'verbose', 'cl_plugin_init')
    (cl_plugins, errors) = load_plugins(app_names, 'cl', conf)
    for (key, val) in errors.items():
        for (plugin_name, load_err) in val:
            logging.error('loading %s %s OpenCL plugin failed : %s'
                          % (key, plugin_name, load_err))
            gpu_exit(conf)
            conf['app_state']['exit_code'] = 1
            return conf['app_state']['exit_code']

    # Prepare configured OpenCL plugins

    hook = 'cl_plugin_init'
    logging.info('Initializing %s OpenCL plugin(s)'
                 % ', '.join([plug[0] for plug in cl_plugins[hook]]))
    req_cl_plugins = cl_plugins.get(hook, [])
    for (name, plugin_mod, args, kwargs) in req_cl_plugins:
        logging.debug('Initialize %s OpenCL plugin' % name)
        try:

            # Always pass conf as first arg

            execute_plugin(hook, name, plugin_mod, [conf] + args,
                           kwargs)
        except Exception:
            logging.error('Init OpenCL plugin %s failed:\n%s' % (name,
                          traceback.format_exc()))
            gpu_exit(conf)
            conf['app_state']['exit_code'] = 1
            return conf['app_state']['exit_code']
    timelog.log(conf, 'verbose', 'cl_plugin_init')

    logging.info('Starting %(detector_shape)s Katsevich reconstruction'
                 % conf)
    log_scan_geometry(conf, opts)
    logging.debug('Full conf: %s' % conf)

    # Start reconstruction

    timelog.set(conf, 'default', 'recon_volume')
    reconstruct_volume(conf, npy_plugins, cl_plugins)
    timelog.log(conf, 'default', 'recon_volume')

    # Get memory usage for logging before cleanup

    total_npy_memory_usage = get_npy_total_size(conf)
    total_cl_memory_usage = get_cl_total_size(conf)

    # Clean up after OpenCL plugins

    timelog.set(conf, 'verbose', 'cl_plugin_exit')
    hook = 'cl_plugin_exit'
    logging.info('Cleaning up after %s OpenCL plugin(s)'
                 % ', '.join([plug[0] for plug in cl_plugins[hook]]))
    req_cl_plugins = cl_plugins.get(hook, [])
    for (name, plugin_mod, args, kwargs) in req_cl_plugins:
        logging.debug('Clean up %s OpenCL plugin' % name)
        try:

            # Always pass conf as first arg

            execute_plugin(hook, name, plugin_mod, [conf] + args,
                           kwargs)
        except Exception:
            logging.error('Exit OpenCL plugin %s failed:\n%s' % (name,
                          traceback.format_exc()))
            gpu_exit(conf)
            conf['app_state']['exit_code'] = 1
            return conf['app_state']['exit_code']
    timelog.log(conf, 'verbose', 'cl_plugin_exit')

    # Clean up after NumPy plugins

    timelog.set(conf, 'verbose', 'npy_plugin_exit')
    hook = 'npy_plugin_exit'
    logging.info('Cleaning up after %s NumPy plugin(s)'
                 % ', '.join([plug[0] for plug in npy_plugins[hook]]))
    req_npy_plugins = npy_plugins.get(hook, [])
    for (name, plugin_mod, args, kwargs) in req_npy_plugins:
        logging.debug('Clean up %s NumPy plugin' % name)
        try:

            # Always pass conf as first arg

            execute_plugin(hook, name, plugin_mod, [conf] + args,
                           kwargs)
        except Exception:
            logging.error('Exit NumPy plugin %s failed:\n%s' % (name,
                          traceback.format_exc()))
            gpu_exit(conf)
            conf['app_state']['exit_code'] = 1
            return conf['app_state']['exit_code']
    timelog.log(conf, 'verbose', 'npy_plugin_exit')

    # Clean up OpenCL memory
    # GC automatically frees all unreachable GPUArray and mem_alloc memory

    timelog.set(conf, 'verbose', 'cl_memory_clean')
    del conf['app_state']['gpu']['prepared_kernels']
    del conf['cl_kernels']
    cl_free_all(conf)
    timelog.log(conf, 'verbose', 'cl_memory_clean')

    (gpu_free, gpu_total) = gpu_mem_info(conf)

    # Don't access GPU after this point

    timelog.set(conf, 'verbose', 'gpu_exit')
    gpu_exit(conf)
    timelog.log(conf, 'verbose', 'gpu_exit')

    logging.info('GPU leaving with %db of dev memory allocated'
                 % get_cl_total_size(conf))
    logging.info('GPU memory: %dMB (%dB) free of %dMB (%dB) total'
                 % (gpu_free / 1024 ** 2, gpu_free, gpu_total / 1024
                 ** 2, gpu_total))

    # Clean up NumPy memory

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
    logging.info('  GPU:  %dMB (%dB)' % (total_cl_memory_usage / 1024
                 ** 2, total_cl_memory_usage))

    if conf['timelog'] == 'verbose':
        logging.info('Init times:')
        logging.info('  conf:                  %.4fs'
                     % timelog.get(conf, 'verbose', 'conf_init'))
        logging.info('  GPU device:            %.4fs'
                     % timelog.get(conf, 'verbose', 'gpu_init'))
        logging.info('  GPU kernels:           %.4fs'
                     % timelog.get(conf, 'verbose', 'gpu_kernel'))
        logging.info('  NumPy plugins:         %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_plugin_init'))
        logging.info('  OpenCL plugins:        %.4fs'
                     % timelog.get(conf, 'verbose', 'cl_plugin_init'))

    logging.info('IO times:')
    logging.info('  load projections:      %.4fs' % timelog.get(conf,
                 'default', 'proj_load'))
    logging.info('  save recon chunks:     %.4fs' % timelog.get(conf,
                 'default', 'recon_save'))

    if conf['timelog'] == 'verbose':
        logging.info('  save projections:      %.4fs'
                     % timelog.get(conf, 'verbose', 'proj_save'))
        logging.info('  transfers to GPU:      %.4fs'
                     % timelog.get(conf, 'verbose', 'host_to_gpu'))
        logging.info('  transfers from GPU:    %.4fs'
                     % timelog.get(conf, 'verbose', 'gpu_to_host'))
        logging.info('  reset GPU data:        %.4fs'
                     % timelog.get(conf, 'verbose', 'memset_gpu'))
        logging.info('GPU kernel times:')
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
    logging.info('  NumPy preprocess:      %.4fs' % timelog.get(conf,
                 'default', 'npy_preprocess'))
    logging.info('  NumPy postprocess:     %.4fs' % timelog.get(conf,
                 'default', 'npy_postprocess'))
    logging.info('  OpenCL preprocess:     %.4fs' % timelog.get(conf,
                 'default', 'cl_preprocess'))
    logging.info('  OpenCL postprocess:    %.4fs' % timelog.get(conf,
                 'default', 'cl_postprocess'))

    if conf['timelog'] == 'verbose':
        logging.info('Cleanup times:')
        logging.info('  NumPy memory:          %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_memory_clean'))
        logging.info('  OpenCL memory:         %.4fs'
                     % timelog.get(conf, 'verbose', 'cl_memory_clean'))
        logging.info('  NumPy plugins:         %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_plugin_exit'))
        logging.info('  OpenCL plugins:        %.4fs'
                     % timelog.get(conf, 'verbose', 'cl_plugin_exit'))
        logging.info('  GPU device:            %.4fs'
                     % timelog.get(conf, 'verbose', 'gpu_exit'))

    logging.info('Complete time used %.3fs' % timelog.get(conf,
                 'default', 'complete'))
    logging.shutdown()
    return conf['app_state']['exit_code']


def usage():
    """Usage help"""

    print 'Usage: %s' % sys.argv[0]
    print 'Run Katsevich reconstruction'


if __name__ == '__main__':
    cl_cfg = default_katsevich_cl_conf()
    cl_opts = default_katsevich_cl_opts()
    try:
        cl_cfg = parse_setup(sys.argv, app_names, cl_opts, cl_cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(2)
    exit_code = main(cl_cfg, cl_opts)
    sys.exit(exit_code)
