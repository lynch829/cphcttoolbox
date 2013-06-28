#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# cucenterfdk - cuda center slice fdk reconstruction
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

"""Circular fan beam CT using the center slice FDK algorithm in cuda"""

import sys
import traceback

from cphct.cu.core import gpu_init_mod, gpu_init_ctx, \
    gpu_kernel_auto_init, gpu_exit, log_gpu_specs, gpu_save_kernels
from cphct.cu.io import cu_free_all, get_cu_data, get_cu_total_size
from cphct.io import create_path_dir
from cphct.log import logging, allowed_log_levels, setup_log, \
    default_level, log_scan_geometry
from cphct.misc import timelog

# These are basic numpy functions exposed through npy to use same numpy

from cphct.npycore.io import get_npy_data, get_npy_total_size, \
    npy_free_all
from cphct.npycore.utils import log_checksum
from cphct.plugins import load_plugins, execute_plugin
from cphct.fan.centerfdk.conf import default_centerfdk_cu_conf, \
    default_centerfdk_cu_opts, parse_setup, ParseError
from cphct.fan.centerfdk.cu.io import fill_centerfdk_cu_conf
from cphct.fan.centerfdk.cu.kernels import init_recon, rt_const, \
    reconstruct_proj

app_names = ['centerfdk', 'cucenterfdk']


def reconstruct_volume(
    conf,
    fdt,
    npy_plugins,
    cu_plugins,
    ):
    """Reconstruct 3D volume from the recorded 2D projections

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    fdt : dtype
        Float data type (internal precision)
    npy_plugins : dict
        A dictionary of numpy plugins to use
    cu_plugins : dict
        A dictionary of CUDA plugins to use

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

    # Initialize GPU proj data transfer structures

    gpu_chunk_index = get_cu_data(conf, 'chunk_index')
    gpu_recon_chunk = get_cu_data(conf, 'recon_chunk')

    gpu_projs_data = get_cu_data(conf, 'projs_data')

    # Get gpu module handle

    gpu_module = conf['gpu']['module']

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

            # Tell GPU which chunk we are processing

            gpu_module.memset_d32(gpu_chunk_index.gpudata, chunk, 1)

            z_voxels_start = chunk * conf['chunk_size']
            z_voxels_end = z_voxels_start + conf['chunk_size']

            logging.info('Reconstructing chunk: %d, z-voxel: %s -> %s'
                         % (chunk, z_voxels_start, z_voxels_end))

            # Reset GPU recon chunk data

            timelog.set(conf, 'verbose', 'memset_gpu', barrier=True)
            gpu_module.memset_d8(gpu_recon_chunk.gpudata, 0,
                                 gpu_recon_chunk.nbytes)
            timelog.log(conf, 'verbose', 'memset_gpu', barrier=True)

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

                # Transfer projection to GPU

                logging.debug('copy %d (%db) projs from host to dev'
                              % (gpu_projs_data.shape[0],
                              gpu_projs_data.nbytes))

                timelog.set(conf, 'verbose', 'host_to_gpu',
                            barrier=True)

                gpu_projs_data.set(projs_data)
                timelog.log(conf, 'verbose', 'host_to_gpu',
                            barrier=True)

                # Preprocess current chunk of projections
                # with configured CUDA plugins

                hook = 'cu_preprocess_input'
                logging.info('Preprocessing with %s cuda plugin(s)'
                             % ', '.join([plug[0] for plug in
                             cu_plugins[hook]]))

                req_cu_plugins = cu_plugins.get(hook, [])

                for (name, plugin_mod, args, kwargs) in req_cu_plugins:
                    logging.debug('Preprocessing chunk with %s cuda plugin'
                                   % name)
                    try:
                        timelog.set(conf, 'default', 'cu_preprocess',
                                    barrier=True)

                        # Always pass gpuarray_filter_in and conf as first args

                        execute_plugin(hook, name, plugin_mod,
                                [gpu_projs_data, projs_meta, conf]
                                + args, kwargs)

                        timelog.log(conf, 'default', 'cu_preprocess',
                                    barrier=True)
                    except Exception:
                        logging.error('Preprocess cuda plugin %s failed:\n%s'
                                 % (name, traceback.format_exc()))
                        gpu_exit(conf)
                        sys.exit(1)

                for load_index in xrange(len(projs_meta)):
                    conf['app_state']['backproject']['proj_idx'] = \
                        proj_index + load_index

                    proj_meta = projs_meta[load_index]

                    # Reconstruct the loaded projection

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

            # Postprocess current chunk of results with configured CUDA plugins

            hook = 'cu_postprocess_output'

            logging.info('Postprocessing with %s cuda plugin(s)'
                         % ', '.join([plug[0] for plug in
                         cu_plugins[hook]]))

            req_cu_plugins = cu_plugins.get(hook, [])
            for (name, plugin_mod, args, kwargs) in req_cu_plugins:
                logging.debug('Postprocessing chunk with %s cuda plugin'
                               % name)
                try:
                    timelog.set(conf, 'default', 'cu_postprocess',
                                barrier=True)

                    # Always pass gpu_output_chunk and conf as first args

                    execute_plugin(hook, name, plugin_mod,
                                   [gpu_recon_chunk, recon_meta, conf]
                                   + args, kwargs)

                    timelog.log(conf, 'default', 'cu_postprocess',
                                barrier=True)
                except Exception:
                    logging.error('Postprocess cuda plugin %s failed:\n%s'
                                   % (name, traceback.format_exc()))
                    gpu_exit(conf)
                    sys.exit(1)

            # Retrieve reconstructed chunk from GPU

            timelog.set(conf, 'verbose', 'gpu_to_host', barrier=True)
            gpu_recon_chunk.get(ary=recon_chunk)
            timelog.log(conf, 'verbose', 'gpu_to_host', barrier=True)

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
        'cu_preprocess',
        'cu_postprocess',
        ]
    verbose = [
        'conf_init',
        'npy_plugin_init',
        'cu_plugin_init',
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
        'cu_plugin_exit',
        'npy_plugin_exit',
        'cu_memory_clean',
        'npy_memory_clean',
        'gpu_exit',
        ]
    timelog.init(conf, default, verbose)

    timelog.set(conf, 'default', 'complete')

    # Complete configuration initialization

    timelog.set(conf, 'verbose', 'conf_init')
    fill_centerfdk_cu_conf(conf)
    fdt = conf['data_type']
    timelog.log(conf, 'verbose', 'conf_init')

    # Init GPU access

    logging.info('Init GPU %d' % conf['cuda_device_index'])

    timelog.set(conf, 'verbose', 'gpu_init')
    gpu_init_mod(conf)
    gpu_module = conf['gpu']['module']
    gpu_count = gpu_module.Device.count()
    if gpu_count < 1:
        logging.error('No GPUs available!')
        sys.exit(1)
    if conf['cuda_device_index'] > gpu_count - 1:
        logging.error('cuda device index (%d) must match GPU IDs! (%s)'
                      % (conf['cuda_device_index'], ', '.join(['%s' % i
                      for i in range(gpu_count)])))
        sys.exit(1)
    if conf['cuda_device_index'] < 0:

        # Just default to the first one for now - 'best' would be better

        conf['cuda_device_index'] = 0

    gpu_init_ctx(conf)
    timelog.log(conf, 'verbose', 'gpu_init')

    log_gpu_specs(conf)
    (gpu_free, gpu_total) = gpu_module.mem_get_info()
    logging.info('GPU memory: %dMB (%dB) free of %dMB (%dB) total'
                 % (gpu_free / 1024 ** 2, gpu_free, gpu_total / 1024
                 ** 2, gpu_total))

    timelog.set(conf, 'verbose', 'gpu_kernel')
    gpu_kernel_auto_init(conf, rt_const)
    gpu_save_kernels(conf)
    timelog.log(conf, 'verbose', 'gpu_kernel')

    if not conf['cu_kernels']:
        logging.error('no valid gpu compute kernels found!')
        gpu_exit(conf)
        sys.exit(1)

    # Initialize FDK kernel data structures

    init_recon(conf, fdt)

    # Load numpy plugins

    timelog.set(conf, 'verbose', 'npy_plugin_init')
    (npy_plugins, errors) = load_plugins(app_names, 'npy', conf)
    for (key, val) in errors.items():
        for (plugin_name, load_err) in val:
            logging.error('loading %s %s numpy plugin failed : %s'
                          % (key, plugin_name, load_err))
            gpu_exit(conf)
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
            gpu_exit(conf)
            sys.exit(1)
    timelog.log(conf, 'verbose', 'npy_plugin_init')

    # Load cuda plugins

    timelog.set(conf, 'verbose', 'cu_plugin_init')
    (cu_plugins, errors) = load_plugins(app_names, 'cu', conf)
    for (key, val) in errors.items():
        for (plugin_name, load_err) in val:
            logging.error('loading %s %s cuda plugin failed : %s'
                          % (key, plugin_name, load_err))
            gpu_exit(conf)
            sys.exit(1)

    # Prepare configured cuda plugins

    hook = 'cu_plugin_init'
    logging.info('Initializing %s cuda plugin(s)' % ', '.join([plug[0]
                 for plug in cu_plugins[hook]]))
    req_cu_plugins = cu_plugins.get(hook, [])
    for (name, plugin_mod, args, kwargs) in req_cu_plugins:
        logging.debug('Initialize %s cuda plugin' % name)
        try:

            # Always pass conf as first arg

            execute_plugin(hook, name, plugin_mod, [conf] + args,
                           kwargs)
        except Exception:
            logging.error('Init cuda plugin %s failed:\n%s' % (name,
                          traceback.format_exc()))
            gpu_exit(conf)
            sys.exit(1)
    timelog.log(conf, 'verbose', 'cu_plugin_init')

    # Start total reconstruction timer

    logging.info('Starting %(detector_shape)s FDK reconstruction'
                 % conf)
    log_scan_geometry(conf, opts)
    logging.debug('Full conf: %s' % conf)

    # Start reconstruction

    timelog.set(conf, 'default', 'recon_volume')
    reconstruct_volume(conf, fdt, npy_plugins, cu_plugins)
    timelog.log(conf, 'default', 'recon_volume')

    # Get memory usage for logging before cleanup

    total_npy_memory_usage = get_npy_total_size(conf)
    total_cu_memory_usage = get_cu_total_size(conf)

    # Clean up after cuda plugins

    timelog.set(conf, 'verbose', 'cu_plugin_exit')
    hook = 'cu_plugin_exit'
    logging.info('Cleaning up after %s cuda plugin(s)'
                 % ', '.join([plug[0] for plug in cu_plugins[hook]]))
    req_cu_plugins = cu_plugins.get(hook, [])
    for (name, plugin_mod, args, kwargs) in req_cu_plugins:
        logging.debug('Clean up %s cuda plugin' % name)
        try:

            # Always pass conf as first arg

            execute_plugin(hook, name, plugin_mod, [conf] + args,
                           kwargs)
        except Exception:
            logging.error('Exit cuda plugin %s failed:\n%s' % (name,
                          traceback.format_exc()))
            gpu_exit(conf)
            sys.exit(1)
    timelog.log(conf, 'verbose', 'cu_plugin_exit')

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
            gpu_exit(conf)
            sys.exit(1)
    timelog.log(conf, 'verbose', 'npy_plugin_exit')

    # Clean up cuda memory

    timelog.set(conf, 'verbose', 'cu_memory_clean')
    cu_free_all(conf)
    timelog.log(conf, 'verbose', 'cu_memory_clean')

    (gpu_free, gpu_total) = gpu_module.mem_get_info()

    # Don't access GPU after this point

    timelog.set(conf, 'verbose', 'gpu_exit')
    gpu_exit(conf)
    timelog.log(conf, 'verbose', 'gpu_exit')

    logging.info('GPU leaving with %db of dev memory allocated'
                 % get_cu_total_size(conf))
    logging.info('GPU memory: %dMB (%dB) free of %dMB (%dB) total'
                 % (gpu_free / 1024 ** 2, gpu_free, gpu_total / 1024
                 ** 2, gpu_total))

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
    logging.info('  gpu:  %dMB (%dB)' % (total_cu_memory_usage / 1024
                 ** 2, total_cu_memory_usage))

    if conf['timelog'] == 'verbose':
        logging.info('Init times:')
        logging.info('  conf:                  %.4fs'
                     % timelog.get(conf, 'verbose', 'conf_init'))
        logging.info('  gpu device:            %.4fs'
                     % timelog.get(conf, 'verbose', 'gpu_init'))
        logging.info('  gpu kernels:           %.4fs'
                     % timelog.get(conf, 'verbose', 'gpu_kernel'))
        logging.info('  numpy plugins:         %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_plugin_init'))
        logging.info('  cuda  plugins:         %.4fs'
                     % timelog.get(conf, 'verbose', 'cu_plugin_init'))

    logging.info('IO times:')
    logging.info('  load projections:      %.4fs' % timelog.get(conf,
                 'default', 'proj_load'))
    logging.info('  save recon chunks:     %.4fs' % timelog.get(conf,
                 'default', 'recon_save'))

    if conf['timelog'] == 'verbose':
        logging.info('  save projections:      %.4fs'
                     % timelog.get(conf, 'verbose', 'proj_save'))
        logging.info('  transfers to gpu:      %.4fs'
                     % timelog.get(conf, 'verbose', 'host_to_gpu'))

        logging.info('  transfers from gpu:    %.4fs'
                     % timelog.get(conf, 'verbose', 'gpu_to_host'))

        logging.info('  Reset gpu data:        %.4fs'
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
    logging.info('  numpy preprocess:      %.4fs' % timelog.get(conf,
                 'default', 'npy_preprocess'))
    logging.info('  numpy postprocess:     %.4fs' % timelog.get(conf,
                 'default', 'npy_postprocess'))
    logging.info('  cuda preprocess:       %.4fs' % timelog.get(conf,
                 'default', 'cu_preprocess'))
    logging.info('  cuda postprocess:      %.4fs' % timelog.get(conf,
                 'default', 'cu_postprocess'))

    if conf['timelog'] == 'verbose':
        logging.info('Cleanup times:')
        logging.info('  numpy memory:          %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_memory_clean'))
        logging.info('  gpu memory:            %.4fs'
                     % timelog.get(conf, 'verbose', 'cu_memory_clean'))
        logging.info('  numpy plugins:         %.4fs'
                     % timelog.get(conf, 'verbose', 'npy_plugin_exit'))
        logging.info('  cuda plugins:          %.4fs'
                     % timelog.get(conf, 'verbose', 'cu_plugin_exit'))
        logging.info('  gpu device:            %.4fs'
                     % timelog.get(conf, 'verbose', 'gpu_exit'))

    logging.info('Complete time used %.3fs' % timelog.get(conf,
                 'default', 'complete'))
    logging.shutdown()


def usage():
    """Usage help"""

    print 'Usage: %s' % sys.argv[0]
    print 'Run center slice FDK reconstruction'


if __name__ == '__main__':
    cu_cfg = default_centerfdk_cu_conf()
    cu_opts = default_centerfdk_cu_opts()
    try:
        cu_cfg = parse_setup(sys.argv, app_names, cu_opts, cu_cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(1)
    main(cu_cfg, cu_opts)
