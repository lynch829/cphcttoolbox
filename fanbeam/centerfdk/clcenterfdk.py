#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# cucenterfdk - OpenCL center slice fdk reconstruction
# Copyright (C) 2012-2014  The CT-Toolbox Project lead by Brian Vinter
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

"""Circular fan beam CT using the center slice FDK algorithm in OpenCL"""

import sys
import traceback

from cphct.cl.core import gpu_init_mod, gpu_init_ctx, gpu_device_count, \
    gpu_kernels_auto_init, gpu_exit, log_gpu_specs, gpu_save_kernels, \
    gpu_mem_info
from cphct.cl.io import cl_free_all, get_cl_data, get_cl_total_size
from cphct.io import create_path_dir
from cphct.log import logging, allowed_log_levels, setup_log, \
    default_level, log_scan_geometry
from cphct.misc import timelog

# These are basic NumPy functions exposed through npy to use same NumPy

from cphct.npycore.io import get_npy_data, get_npy_total_size, \
    npy_free_all
from cphct.npycore.utils import log_checksum
from cphct.plugins import load_plugins, execute_plugin
from cphct.fan.centerfdk.conf import default_centerfdk_cl_conf, \
    default_centerfdk_cl_opts, parse_setup, ParseError
from cphct.fan.centerfdk.cl.io import fill_centerfdk_cl_conf
from cphct.fan.centerfdk.cl.kernels import init_recon, rt_const, \
    reconstruct_proj

app_names = ['centerfdk', 'clcenterfdk']


def __move_projs_to_GPU(
    conf,
    gpu_projs_data,
    projs_data,
    projs_meta,
    ):
    """Moves a boundingboxed projection chunk to the GPU
    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    gpu_projs_data : gpuarray
        GPU projection data array
    projs_data : ndarray
        CPU projection data array
    projs_meta : list of dict
        List of meta data dictionaries matching projs_data
    
    Returns
    -------
    output : gpuarray
        Returns gpu array with boundingboxed projection data
    """

    gpu_queue = conf['gpu']['queue']
    start_row = conf['app_state']['projs']['boundingbox'][0, 0]
    end_row = conf['app_state']['projs']['boundingbox'][0, 1]
    row_count = end_row - start_row

    for proj_index in xrange(len(projs_meta)):
        logging.debug('copy proj htod: %d, start_row: %d, end_row: %d'
                      % (proj_index, start_row, end_row))

        timelog.set(conf, 'verbose', 'host_to_gpu', barrier=True)

        gpu_projs_data[proj_index, :
                       row_count].set(projs_data[proj_index, start_row:
                end_row], queue=gpu_queue)
        timelog.log(conf, 'verbose', 'host_to_gpu', barrier=True)

    return gpu_projs_data


def reconstruct_volume(
    conf,
    fdt,
    npy_plugins,
    cl_plugins,
    ):
    """Reconstruct 3D volume from the recorded 2D projections

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    fdt : dtype
        Float data type (internal precision)
    npy_plugins : dict
        A dictionary of NumPy plugins to use
    cl_plugins : dict
        A dictionary of OpenCL plugins to use

    Returns
    -------
    output : dict
        Dictionary of configuration options with updated timelog
    """

    # Create save path dirs and empty binary save files

    if conf['save_filtered_projs_data_path']:
        create_path_dir(conf['save_filtered_projs_data_path'])
        open(conf['save_filtered_projs_data_path'], 'wb', 0).close()

    # Init projection and recon meta data lists

    (projs_meta, recon_meta) = ([], [])

    # Get pre-allocated projection and recon chunk data

    projs_data = get_npy_data(conf, 'projs_data')
    recon_chunk = get_npy_data(conf, 'recon_chunk')
    detector_boundingboxes = get_npy_data(conf, 'detector_boundingboxes'
            )

    # Get GPU data structures

    gpu_recon_chunk = get_cl_data(conf, 'recon_chunk')
    gpu_projs_data = get_cl_data(conf, 'projs_data')

    # Get GPU queue handle

    gpu_queue = conf['gpu']['queue']

    # Loop over the amount of steps in the scene

    for step in xrange(conf['total_turns']):
        logging.info('Reconstructing step: %s/%s' % (step,
                     conf['total_turns']))

        # Set current step

        conf['app_state']['scanner_step'] = step

        # Loop over the amount of volume chunks in the z direction

        for chunk in xrange(conf['chunk_count']):
            if not chunk in conf['chunks_enabled']:
                continue

            # Set current chunk

            conf['app_state']['chunk']['idx'] = chunk

            # Set projection bounding box for the current chunk

            conf['app_state']['projs']['boundingbox'] = \
                detector_boundingboxes[chunk]

            z_voxels_start = chunk * conf['chunk_size']
            z_voxels_end = z_voxels_start + conf['chunk_size']

            logging.info('Reconstructing chunk: %d, z-voxel: %s -> %s'
                         % (chunk, z_voxels_start, z_voxels_end))

            # Reset GPU recon chunk data
            # NOTE: Due to memory leak in pyopencl.array.Array.fill
            #       we call .finish to cleanup in each iteration

            timelog.set(conf, 'verbose', 'memset_gpu', barrier=True)
            gpu_recon_chunk.finish()
            gpu_recon_chunk.fill(0, queue=gpu_queue)
            timelog.log(conf, 'verbose', 'memset_gpu', barrier=True)

            # Loop over the projections in each chunk for each step

            for proj in xrange(0, conf['projs_per_turn'],
                               conf['proj_chunk_size']):

                # Load next projection chunk

                proj_index = step * conf['projs_per_turn'] + proj

                # Set the projection range we are processing

                conf['app_state']['projs']['first'] = proj_index
                conf['app_state']['projs']['last'] = min(proj_index
                        + conf['proj_chunk_size'], conf['projs_per_turn'
                        ]) - 1

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
                        conf['app_state']['exit_code'] = 1
                        return conf

                if conf['checksum']:
                    chunk_view = projs_data.ravel()
                    log_checksum('Raw projs chunk', chunk_view,
                                 chunk_view.size)

                # Preprocess current chunk of projections
                # with configured NumPy plugins

                hook = 'npy_preprocess_input'
                req_npy_plugins = npy_plugins.get(hook, [])

                logging.info('Preprocessing with %s NumPy plugin(s)'
                             % ', '.join([plug[0] for plug in
                             npy_plugins[hook]]))

                for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                    logging.debug('Preprocessing chunk with %s NumPy plugin'
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
                        logging.error('Preprocess NumPy plugin %s failed:\n%s'
                                 % (name, traceback.format_exc()))
                        conf['app_state']['exit_code'] = 1
                        return conf

                if conf['checksum']:
                    chunk_view = projs_data.ravel()
                    log_checksum('Preprocessed projs chunk',
                                 chunk_view, chunk_view.size)

                # Transfer projection to GPU

                __move_projs_to_GPU(conf, gpu_projs_data, projs_data,
                                    projs_meta)

                # Preprocess current chunk of projections
                # with configured OpenCL plugins

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

                        # Always pass gpuarray_filter_in and conf as first args

                        execute_plugin(hook, name, plugin_mod,
                                [gpu_projs_data, projs_meta, conf]
                                + args, kwargs)

                        timelog.log(conf, 'default', 'cl_preprocess',
                                    barrier=True)
                    except Exception:
                        logging.error('Preprocess OpenCL plugin %s failed:\n%s'
                                 % (name, traceback.format_exc()))
                        gpu_exit(conf)
                        conf['app_state']['exit_code'] = 1
                        return conf

                for load_index in xrange(len(projs_meta)):
                    conf['app_state']['backproject']['proj_idx'] = \
                        proj_index + load_index

                    proj_meta = projs_meta[load_index]

                    # Reconstruct the preprocessed projection

                    timelog.set(conf, 'verbose', 'proj_recon',
                                barrier=True)

                    reconstruct_proj(conf, proj_meta, fdt)

                    log_time = timelog.log(conf, 'verbose', 'proj_recon'
                            , barrier=True)

                    msg = 'Reconstructed projection: %s, angle: %s' \
                        % (conf['app_state']['backproject']['proj_idx'
                           ], proj_meta['angle'])

                    if conf['timelog'] == 'verbose':
                        msg = '%s in %.4f seconds' % (msg, log_time)

                    logging.info(msg)

                    if conf['checksum']:
                        gpu_recon_chunk.get(ary=recon_chunk)
                        recon_chunk_view = recon_chunk.ravel()
                        log_checksum('Recon chunk', recon_chunk_view,
                                recon_chunk_view.size)

            # Postprocess current result chunk with configured OpenCL plugins

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

                    # Always pass gpu_output_chunk and conf as first args

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

            # Retrieve reconstructed chunk from GPU

            timelog.set(conf, 'verbose', 'gpu_to_host', barrier=True)
            gpu_recon_chunk.get(ary=recon_chunk)
            timelog.log(conf, 'verbose', 'gpu_to_host', barrier=True)

            # Postprocess current chunk of results with configured NumPy plugins

            hook = 'npy_postprocess_output'
            req_npy_plugins = npy_plugins.get(hook, [])
            logging.info('Postprocessing with %s NumPy plugin(s)'
                         % ', '.join([plug[0] for plug in
                         npy_plugins[hook]]))

            for (name, plugin_mod, args, kwargs) in req_npy_plugins:
                logging.debug('Postprocessing chunk with %s NumPy plugin'
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
                    logging.error('Postprocess NumPy plugin %s failed:\n%s'
                                   % (name, traceback.format_exc()))
                    conf['app_state']['exit_code'] = 1
                    return conf

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
                    conf['app_state']['exit_code'] = 1
                    return conf

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

    # Initialize timelog

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
    verbose = [
        'conf_init',
        'npy_plugin_init',
        'cl_plugin_init',
        'gpu_init',
        'gpu_kernel',
        'memset_gpu',
        'host_to_gpu',
        'proj_weight',
        'proj_filter',
        'proj_save',
        'transform_matrix',
        'volume_weight',
        'backproject',
        'proj_recon',
        'gpu_to_host',
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
    fill_centerfdk_cl_conf(conf)
    fdt = conf['data_type']
    timelog.log(conf, 'verbose', 'conf_init')

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

    # Initialize FDK kernel data structures

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

    # Start total reconstruction timer

    logging.info('Starting %(detector_shape)s FDK reconstruction'
                 % conf)
    log_scan_geometry(conf, opts)
    logging.debug('Full conf: %s' % conf)

    # Start reconstruction

    timelog.set(conf, 'default', 'recon_volume')
    reconstruct_volume(conf, fdt, npy_plugins, cl_plugins)
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

    timelog.set(conf, 'verbose', 'cl_memory_clean')
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

    if conf['timelog'] == 'verbose':
        logging.info('GPU recon kernels: %.4fs' % timelog.get(conf,
                     'verbose', 'proj_recon'))
        logging.info('(proj. avg. %.4fs) (chunk avg. %.4fs)'
                     % (timelog.get(conf, 'verbose', 'proj_recon')
                     / conf['total_projs'], timelog.get(conf, 'verbose'
                     , 'proj_recon') / conf['chunk_count']))

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
        logging.info('  OpenCL  plugins:         %.4fs'
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

        logging.info('  Reset GPU data:        %.4fs'
                     % timelog.get(conf, 'verbose', 'memset_gpu'))
        logging.info('GPU kernel times:')
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
    logging.info('  NumPy preprocess:      %.4fs' % timelog.get(conf,
                 'default', 'npy_preprocess'))
    logging.info('  NumPy postprocess:     %.4fs' % timelog.get(conf,
                 'default', 'npy_postprocess'))
    logging.info('  OpenCL preprocess:       %.4fs' % timelog.get(conf,
                 'default', 'cl_preprocess'))
    logging.info('  OpenCL postprocess:      %.4fs' % timelog.get(conf,
                 'default', 'cl_postprocess'))

    if conf['timelog'] == 'verbose':
        logging.info('Cleanup times:')
        logging.info('  NumPy memory:          %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_memory_clean'))
        logging.info('  GPU memory:            %.4fs'
                     % timelog.get(conf, 'verbose', 'cl_memory_clean'))
        logging.info('  NumPy plugins:         %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_plugin_exit'))
        logging.info('  OpenCL plugins:          %.4fs'
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
    print 'Run center slice FDK reconstruction'


if __name__ == '__main__':
    cl_cfg = default_centerfdk_cl_conf()
    cl_opts = default_centerfdk_cl_opts()
    try:
        cl_cfg = parse_setup(sys.argv, app_names, cl_opts, cl_cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(2)
    exit_code = main(cl_cfg, cl_opts)
    sys.exit(exit_code)
