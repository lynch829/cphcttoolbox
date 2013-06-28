#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# conf - configuration helpers
# Copyright (C) 2011-2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Configuration helper functions"""

import ConfigParser
import StringIO
import getopt
import os
import sys
import time
import tempfile
from math import pi

from cphct import package_version
from cphct.io import path_expander
from cphct.misc import nextpow2
from cphct.misc.timelog import allowed_timelogs

# Valid values for string options:
# In case of different public and internal names we use a dictionary mapping
# the public name to the internal name. Otherwise we just use a shared list of
# strings.

# TODO: enable opencl once it is supported
# allowed_engines = {'numpy': 'npy', 'cuda': 'cu', 'opencl': 'ocl'}

allowed_engines = {'numpy': 'npy', 'cuda': 'cu'}
allowed_scanning_paths = ['step', 'helix']
allowed_detector_shapes = ['flat', 'curved']
allowed_interpolation = ['point', 'linear']

default_engine = 'numpy'


def any_value(val):
    """Always returns the raw input value

    Parameters
    ----------
    val : object
        Any value from the configuration parsing.

    Returns
    -------
    output : object
        Returns the raw input value unchanged.
    """

    return val


def bool_value(val):
    """Force to boolean

    Parameters
    ----------
    val : object
        Any value from the configuration parsing to be cast to a boolean.

    Returns
    -------
    output : bool
        Returns the raw input value cast to a boolean.
    """

    return bool(val)


def int_value(val):
    """Force to integer

    Parameters
    ----------
    val : object
        Any value from the configuration parsing to be cast to an integer.

    Returns
    -------
    output : int
        Returns the raw input value cast to an integer.
    """

    return int(val)


def int_pow2_value(val):
    """Checks if *val* is a power of two

    Parameters
    ----------
    val : str
        Any value to check for if power of two

    Returns
    -------
    output : int
        Returns *val* cast to int if it's a power of two

    Raises
    ------
    ParseError
        If *val* is not a power of two
    """

    val = int_value(val)
    pow2_val = nextpow2(val)
    if val != pow2_val:
        raise ParseError('invalid value \'%s\', must be a power of two'
                         % val)

    return val


def long_value(val):
    """Force to long

    Parameters
    ----------
    val : object
        Any value from the configuration parsing to be cast to a long.

    Returns
    -------
    output : long
        Returns the raw input value cast to a long.
    """

    return long(val)


def float_value(val):
    """Force to float

    Parameters
    ----------
    val : object
        Any value from the configuration parsing to be cast to a float.

    Returns
    -------
    output : float
        Returns the raw input value cast to a float.
    """

    return float(val)


def str_value(val):
    """Force to str

    Parameters
    ----------
    val : object
        Any value from the configuration parsing to be cast to a string.

    Returns
    -------
    output : str
        Returns the raw input value cast to a string.
    """

    return str(val)


def colon_int_values(val):
    """Split string of colon separated values to list of ints

    Parameters
    ----------
    val : str
        A string with colon separated values to be interpreted as a list of
        integers.

    Returns
    -------
    output : list of int
        Returns a list of integers extracted from the string of colon
        separated values.
    """

    return [int(i) for i in str(val).split(':')]


def colon_float_values(val):
    """Split string of colon separated values to list of floats

    Parameters
    ----------
    val : str
        A string with colon separated values to be interpreted as a list of
        floats.

    Returns
    -------
    output : list of float
        Returns a list of floats extracted from the string of colon
        separated values.
    """

    return [float(i) for i in str(val).split(':')]


def colon_str_values(val):
    """Split string of colon separated values to list of strings

    Parameters
    ----------
    val : str
        A string with colon separated values to be interpreted as a list of
        strings.

    Returns
    -------
    output : list of str
        Returns a list of strings extracted from the string of colon
        separated values.
    """

    return str(val).split(':')


def hash_str_values(val):
    """Split string of hash separated values to list of strings

    Parameters
    ----------
    val : str
        A string with hash ('#') separated values to be interpreted as a list
        of strings.

    Returns
    -------
    output : list of str
        Returns a list of strings extracted from the string of hash separated
        values.
    """

    return str(val).split('#')


def named_arg_str_values(val):
    """Split string of equal-sign separated named arg and value to 2-tuple of
    strings.  I.e. ('name', 'value') for the input string 'name=value'.

    Parameters
    ----------
    val : str
        A string with equal-sign ('=') separated values to be interpreted as a
        list of strings.

    Returns
    -------
    output : (str, str)
        Returns a 2-tuple of (key, val) strings extracted from the string of
        equal-sign separated values.
    """

    return tuple([i.strip() for i in str(val).split('=', 1)])


def plugin_arg_values(val):
    """Split plugin argument string of colon separated plugin entries each
    containing plugin name optionally followed by hash mark separated args to
    a list of (name, args, kwargs) tuples. The args and kwargs entries are
    on the typical args list and kwargs dictionary form and the hash separated
    entries can be raw values passed as positional args or named args passed
    as kwargs.
    Examples:
    val='foo:bar#bla' -> [('foo', [], {}), ('bar', ['bla'], {})]
    val='normalize#0#1' -> [('normalize', ['0', '1'], {})]
    val='clip#0#clip_max=1' -> [('clip', ['0'], {'clip_max': '1'})]

    Parameters
    ----------
    val : str
        A string with complex nested argument values to be interpreted as a
        list of plugin argument tuples.

    Returns
    -------
    output : list of tuple
        Returns a list of tuples with each tuple representing a plugin call.
    """

    plugins = []
    plugin_entries = colon_str_values(val)
    for entry in plugin_entries:
        plugin_parts = hash_str_values(entry)
        (name, args, kwargs) = (plugin_parts[0], [], {})
        if not name:
            continue
        for plugin_arg in plugin_parts[1:]:
            named_arg = named_arg_str_values(plugin_arg)
            if named_arg[1:]:
                kwargs[named_arg[0]] = named_arg[1]
            else:
                args.append(named_arg[0])
        plugins.append((name, args, kwargs))
    return plugins


def gap_pair_values(val):
    """Split gap position string of comma seperated entries each
    containing a comma separated range to a list of gap (start, end) tuples
    where start and end are floating point numbers.
    Examples:
    val='(-0.1,0.1) -> [(-0.1, 0.1)]
    val='(-12.2,- 10.0),(-1.1, 1.1),(10.0,12.2) -> [(-12.2, 10.0),(-1.1, 1.1),
                                              (10.0, 12.2)]

    Parameters
    ----------
    val : str
        A string with complex nested argument values to be interpreted as a
        list of detector gap position tuples.

    Returns
    -------
    output : list of tuple
        Returns a list of tuples with each tuple representing a detector gap
        position.
    """

    output = []
    gap_pair_str = val[1:-1].split('),(')
    for gap_pair in gap_pair_str:
        (start, end) = gap_pair.split(',')
        gap_pair_tuple = (float(start), float(end))
        output.append(gap_pair_tuple)

    return output


def allowed_values(val, allowed):
    """Checks if *val* is in *allowed* values

    Parameters
    ----------
    val : str
        Any value to check against allowed
    allowed : list of str or dict
        Values allowed for *val*. Use entries if list and keys if dict.

    Returns
    -------
    output : str
        Returns *val* if it's a valid entry or key in *allowed*

    Raises
    ------
    ParseError
        If *val* is not in *allowed*
    """

    if not val in allowed:
        raise ParseError('invalid value \'%s\', allowed values: %s'
                         % (val, ', '.join([i for i in allowed])))

    return val


def usage(argv, opt_handlers, conf):
    """Print usage information based on opt_handlers

    Parameters
    ----------
    argv : list of str
        The list of call arguments.
    opt_handlers : dict
        A dictionary of application option specs used for parsing and doc.
    conf : dict
        A dictionary of configuration options.
    """

    print 'USAGE: %s [OPTIONS] ARGS' % argv[0]
    print 'where OPTIONS may include the following:'
    for (conf_opt, handler) in opt_handlers.items():
        short_opt = handler.get('short', None)
        long_opt = handler.get('long', None)
        use_line = ''

        # Skip internal helpers with neither short nor long option

        if long_opt and short_opt:
            use_line += '--%(long)s / -%(short)s '
        elif long_opt:
            use_line += '--%(long)s '
        elif short_opt:
            use_line += '-%(short)s '
        else:
            continue
        if handler.get('args', None):
            use_line += '/ cfg:%s' % conf_opt
            use_line += '    %(args)s'
        else:
            use_line += ''
        value_help_str = ''
        if handler.get('value_help', None) is not None:
            value_help_str = ' (%(value_help)s)' % handler
        default_str = ''
        if handler.get('default', None) is not None:
            default_str = '\n    Default: '
            default_value = handler['default']
            if not isinstance(default_value, basestring) \
                and isinstance(default_value, list):
                default_str += '%s' % ', '.join(default_value)
            else:
                default_str += '%s' % default_value
        print use_line % handler
        allowed_str = ''
        if handler.get('allowed', None) is not None:
            allowed_str = '\n    Allowed: '
            allowed_value = handler['allowed']
            if not isinstance(allowed_value, basestring) \
                and isinstance(allowed_value, list):
                allowed_str += '%s' % ', '.join(allowed_value)
            else:
                allowed_str += '%s' % allowed_value
        if handler.get('description', None):
            desc_line = '    %(description)s' % handler
            desc_line += value_help_str
            desc_line += default_str
            desc_line += allowed_str
            print desc_line


def version_exit(dummy):
    """Print version and exit cleanly

    Parameters
    ----------
    dummy : object
        Any object that will be ignored.
    """

    print package_version
    sys.exit(0)


def help_exit(
    argv,
    opt_handlers,
    conf,
    dummy,
    ):
    """Print help and exit cleanly

    Parameters
    ----------
    argv : list of str
        The list of call arguments.
    opt_handlers : dict
        A dictionary of application option specs used for parsing and doc.
    conf : dict
        A dictionary of configuration options.
    dummy : object
        Any object that will be ignored.
    """

    usage(argv, opt_handlers, conf)
    sys.exit(0)


def _shared_opts():
    """Return dictionary with base options

    Returns
    -------
    output : dict
        Returns a dictionary of options helper dictionaries.
    """

    # chunk_range defines chunks of FoV to reconstruct in each step to improve
    # memory locality for cache optimizations and to limit total memory
    # requirements. The actual mapping to FoV chunks is application dependent.
    # In the cone beam applications we map chunks to a number of z-slices.

    opts = {
        'args': {
            'long': None,
            'short': None,
            'args': None,
            'handler': None,
            'default': [],
            'description': 'Internal command line argument list',
            },
        'time_stamp': {
            'long': None,
            'short': None,
            'args': None,
            'handler': None,
            'default': time.time(),
            'description': 'Internal time stamp for generated files',
            },
        'arc': {
            'long': None,
            'short': None,
            'args': None,
            'handler': None,
            'default': 2 * pi,
            'description': 'Internal rotation helper',
            },
        'source_count': {
            'long': None,
            'short': None,
            'args': None,
            'handler': None,
            'default': 0,
            'description': 'Internal source position helper',
            },
        'engine': {
            'long': 'engine',
            'short': 'E',
            'args': str,
            'handler': None,
            'default': None,
            'description': 'Compatibility option which is silently ignored',
            },
        'limit_sources': {
            'long': 'limit-sources',
            'short': None,
            'args': str,
            'handler': str_value,
            'default': None,
            'description': 'Limit reconstruction to given source angles',
            },
        'angle_start': {
            'long': 'angle-start',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 0.0,
            'description': 'Which angle to begin from',
            },
        'checksum': {
            'long': 'checksum',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 0,
            'description': 'Level of checksums to enable',
            },
        'detector_shape': {
            'long': 'detector-shape',
            'short': None,
            'args': str,
            'handler': lambda val: allowed_values(val,
                    allowed_detector_shapes),
            'default': allowed_detector_shapes[0],
            'description': 'Shape of detector',
            'allowed': allowed_detector_shapes,
            },
        'detector_pixel_width': {
            'long': 'detector-pixel-width',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1,
            'description': 'Detector pixel width in cm',
            'value_help': '-1 for auto',
            },
        'detector_width': {
            'long': 'detector-width',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1,
            'description': 'Detector width in cm',
            'value_help': '-1 for auto',
            },
        'detector_columns': {
            'long': 'detector-columns',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 256,
            'description': 'Number of pixel columns in projections',
            },
        'detector_column_offset': {
            'long': 'detector-column-offset',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 0.0,
            'description': 'Center ray alignment offset in pixel columns',
            },
        'log_path': {
            'long': 'log-path',
            'short': 'l',
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': '',
            'description': 'Log file path',
            'value_help': 'empty for stdout',
            },
        'log_format': {
            'long': 'log-format',
            'short': None,
            'args': str,
            'handler': str_value,
            'default': None,
            'description': 'Log line format',
            },
        'log_level': {
            'long': 'log-level',
            'short': 'L',
            'args': str,
            'handler': str_value,
            'default': None,
            'description': 'Log verbosity',
            },
        'timelog': {
            'long': 'timelog',
            'short': None,
            'args': str,
            'handler': lambda val: allowed_values(val,
                    allowed_timelogs),
            'default': allowed_timelogs[0],
            'description': 'Log execution times',
            'allowed': allowed_timelogs,
            },
        'precision': {
            'long': 'precision',
            'short': 'p',
            'args': str,
            'handler': str_value,
            'default': 'float32',
            'description': 'Select internal data type (precision)',
            },
        'complex_precision': {
            'long': 'complex-precision',
            'short': None,
            'args': str,
            'handler': str_value,
            'default': 'complex64',
            'description': 'Select internal complex data type (precision)',
            },
        'input_precision': {
            'long': 'input-precision',
            'short': 'i',
            'args': str,
            'handler': str_value,
            'default': 'float32',
            'description': 'Select input data type (precision)',
            },
        'output_precision': {
            'long': 'output-precision',
            'short': None,
            'args': str,
            'handler': str_value,
            'default': 'float32',
            'description': 'Select output data type (precision)',
            },
        'source_distance': {
            'long': 'source-distance',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 3.0,
            'description': 'Distance in cm from source to isocenter',
            },
        'detector_distance': {
            'long': 'detector-distance',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 3.0,
            'description': 'Distance in cm from isocenter to detector',
            },
        'x_min': {
            'long': 'x-min',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1.0,
            'description': 'Field of View minimum x coordinate in cm',
            },
        'x_max': {
            'long': 'x-max',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 1.0,
            'description': 'Field of View maximum x coordinate in cm',
            },
        'y_min': {
            'long': 'y-min',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1.0,
            'description': 'Field of View minimum y coordinate in cm',
            },
        'y_max': {
            'long': 'y-max',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 1.0,
            'description': 'Field of View maximum y coordinate in cm',
            },
        'x_voxels': {
            'long': 'x-voxels',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 512,
            'description': 'Field of View resolution in x',
            },
        'y_voxels': {
            'long': 'y-voxels',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 512,
            'description': 'Field of View resolution in y',
            },
        'projs_per_turn': {
            'long': 'projs-per-turn',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 360,
            'description': 'Number of projections in a full gantry rotation',
            },
        'total_turns': {
            'long': 'total-turns',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': -1,
            'description': 'Number of full gantry rotations',
            'value_help': '-1 for auto',
            },
        'save_filtered_projs_data_path': {
            'long': 'save-filtered-projs-data-path',
            'short': None,
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': '',
            'description': 'Save filtered binary projection data in given ' \
                + 'path',
            },
        'working_directory': {
            'long': 'working-directory',
            'short': 'w',
            'args': str,
            'handler': str_value,
            'default': '.',
            'description': 'Prefix for all relative paths',
            },
        'scanning_path': {
            'long': None,
            'short': None,
            'args': None,
            'handler': None,
            'default': None,
            'description': 'Scanner move path',
            'allowed': allowed_scanning_paths,
            },
        'temporary_directory': {
            'long': 'temporary-directory',
            'short': None,
            'args': str,
            'handler': str_value,
            'default': tempfile.gettempdir(),
            'description': 'Prefix for all temporary file paths',
            },
        'plugin_shared_vars': {
            'long': None,
            'short': None,
            'args': None,
            'handler': None,
            'default': {},
            'description': 'Helper used to share variables between plugins',
            },
        'chunk_range': {
            'long': 'chunk-range',
            'short': None,
            'args': str,
            'handler': colon_int_values,
            'default': None,
            'description': 'Select range of chunks to reconstruct',
            },
        'chunk_size': {
            'long': 'chunk-size',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': -1,
            'description': 'Application dependent size of reconstruction ' + \
            'chunks',
            },
        }

    return opts


def _npycore_opts():
    """Shared numpy core options

    Returns
    -------
    output : dict
        Returns a dictionary of options helper dictionaries for all engines
        using a numpy core.
    """

    opts = {
        'npy_load_input': {
            'long': 'npy-load-input',
            'short': None,
            'args': str,
            'handler': plugin_arg_values,
            'default': [],
            'description': 'Input loader to parse and read input data into ' \
                + 'array',
            },
        'npy_preprocess_input': {
            'long': 'npy-preprocess-input',
            'short': None,
            'args': str,
            'handler': plugin_arg_values,
            'default': [],
            'description': 'Apply specified numpy preprocessing to input ' \
                + 'array',
            },
        'npy_postprocess_output': {
            'long': 'npy-postprocess-output',
            'short': None,
            'args': str,
            'handler': plugin_arg_values,
            'default': [],
            'description': 'Apply specified numpy postprocessing to output ' \
                + 'array',
            },
        'npy_save_output': {
            'long': 'npy-save-output',
            'short': None,
            'args': str,
            'handler': plugin_arg_values,
            'default': [],
            'description': 'Output saver to format and write output data to ' \
                + 'one or more files',
            },
        }

    return opts


def _npy_opts():
    """Numpy specific options

    Returns
    -------
    output : dict
        Returns a dictionary of numpy specific options helper dictionaries.
    """

    opts = {}
    opts.update(_npycore_opts())
    return opts


def _cu_opts():
    """CUDA specific options

    Returns
    -------
    output : dict
        Returns a dictionary of cuda specific options helper dictionaries.
    """

    # gpu_target_threads is a helper to divide work between GPU cores. We have
    # found best performance with multiple of 32 threads per block and many
    # blocks.
    # CUDA Best Practices Guide recommends 128 or 256 threads per block
    # as long as there are enough blocks to easily fill multiprocessor

    # gpu_target_filter_memory is a helper to split up filtering into
    # reasonably small chunks to fit in GPU memory.

    # gpu_target_input_memory is a helper to split up backprojection input into
    # reasonably small chunks to fit in GPU memory.

    # gpu_projs_only is a helper to indicate if projs can be kept on the gpu
    # all the time.
    # Only copy filtered projections back to host for debug and saving.

    cu_opts = {
        'cu_plugin_state': {
            'long': None,
            'short': None,
            'args': None,
            'handler': None,
            'default': {},
            'description': 'State helper for use by cuda plugins',
            },
        'cu_preprocess_input': {
            'long': 'cu-preprocess-input',
            'short': None,
            'args': str,
            'handler': plugin_arg_values,
            'default': [],
            'description': 'Apply specified cuda preprocessing to input array',
            },
        'cu_postprocess_output': {
            'long': 'cu-postprocess-output',
            'short': None,
            'args': str,
            'handler': plugin_arg_values,
            'default': [],
            'description': 'Apply specified cuda postprocessing to output ' \
                + 'array',
            },
        'cuda_device_index': {
            'long': 'cuda-device-index',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': -1,
            'description': 'Which CUDA device to use: -1 for auto',
            },
        'gpu_target_threads': {
            'long': 'gpu-target-threads',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 256,
            'description': 'Number of CUDA kernel threads to aim for per ' \
                + 'block',
            },
        'gpu_target_filter_memory': {
            'long': 'gpu-target-filter-memory',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 256*2**20,
            'description': 'GPU memory in bytes to aim for with projection ' \
                + 'filter chunking',
            },
        'gpu_target_input_memory': {
            'long': 'gpu-target-input-memory',
            'short': None,
            'args': int,
            'handler': int_value,
            'default': 128*2**20,
            'description': 'GPU memory in bytes to aim for as input for ' \
                + 'backprojection chunking',
            },
        'gpu_projs_only': {
            'long': 'gpu-projs-only',
            'short': None,
            'args': bool,
            'handler': bool_value,
            'default': True,
            'description': 'Keep partial results on the GPU - no copy to host',
            },
        'load_gpu_init_path': {
            'long': 'load-gpu-init-path',
            'short': None,
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': '',
            'description': 'Path to load optional CUDA init code from',
            },
        'load_gpu_kernels_path': {
            'long': 'load-gpu-kernels-path',
            'short': None,
            'args': None,
            'handler': None,
            'expander': None,
            'default': None,
            'description': 'Path to load CUDA kernels code from',
            },
        'save_gpu_kernels_path': {
            'long': 'save-gpu-kernels-path',
            'short': None,
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': '',
            'description': 'Path to save runtime optimized CUDA kernels code ' \
                + 'in',
            },
        'load_gpu_binary_path': {
            'long': 'load-gpu-binary-path',
            'short': None,
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': '',
            'description': 'Path to load compiled CUDA kernels from',
            },
        'save_gpu_binary_path': {
            'long': 'save-gpu-binary-path',
            'short': None,
            'args': str,
            'handler': str_value,
            'expander': path_expander,
            'default': '',
            'description': 'Path to save runtime optimized compiled CUDA ' \
                + 'kernels in',
            },
        'host_params_remap': {
            'long': None,
            'short': None,
            'args': float,
            'handler': None,
            'default': [],
            'description': 'List of variables to auto remap in cuda kernel',
            },
        'gpu': {
            'long': None,
            'short': None,
            'args': None,
            'handler': None,
            'default': {},
            'description': 'GPU handle for internal use',
            },
        }

    opts = {}
    opts.update(_npycore_opts())
    opts.update(cu_opts)
    return opts


def _ocl_opts():
    """OpenCL specific options

    Returns
    -------
    output : dict
        Returns a dictionary of opencl specific options helper dictionaries.
    """

    ocl_opts = {}
    opts = {}
    opts.update(_npycore_opts())
    opts.update(ocl_opts)
    return opts


def engine_opts():
    """For enabling engine selection option

    Returns
    -------
    output : dict
        Returns a dictionary of engine options helper dictionaries.
    """

    opts = {'engine': {
        'long': 'engine',
        'short': 'E',
        'args': str,
        'handler': str_value,
        'default': default_engine,
        'allowed': allowed_engines.keys(),
        'description': 'Back end calculation engine',
        }}

    return opts


def default_base_opts():
    """Basic command line option parser dictionary used for option parsing

    Returns
    -------
    output : dict
        Returns a dictionary of base options helper dictionaries.
    """

    # Long and short options supported in all scripts - getopt format
    # All scripts accept engine option, but silently ignores it by default

    return _shared_opts()


def default_base_npy_opts():
    """Numpy specific options

    Returns
    -------
    output : dict
        Returns a dictionary of base numpy options helper dictionaries.
    """

    opts = default_base_opts()
    opts.update(_npy_opts())
    return opts


def default_base_cu_opts():
    """Cuda specific options

    Returns
    -------
    output : dict
        Returns a dictionary of base cuda options helper dictionaries.
    """

    opts = default_base_opts()
    opts.update(_cu_opts())
    return opts


def enable_conf_option(opts):
    """Shared enable conf option helper

    Parameters
    ----------
    opts : dict
        options dictionary.

    Returns
    -------
    output : dict
        Returns a configuration dictionary extracted from options helper
        dictionary.
    """

    conf = {}
    for (key, val) in opts.items():
        conf[key] = val['default']
    return conf


def engine_conf():
    """For enabling option of same name in conf

    Returns
    -------
    output : dict
        Returns a configuration dictionary where engine option is set.
    """

    return enable_conf_option(engine_opts())


def default_base_conf():
    """Initialize base configuration dictionary with default values

    Returns
    -------
    output : dict
        Returns a dictionary of base conf settings.
    """

    conf = {}
    for (key, val) in default_base_opts().items():
        conf[key] = val['default']
    return conf


def default_base_npy_conf():
    """Configuration dictionary with default values for numpy engine

    Returns
    -------
    output : dict
        Returns a dictionary of base numpy conf settings.
    """

    conf = default_base_conf()
    for (key, val) in default_base_npy_opts().items():
        conf[key] = val['default']
    return conf


def default_base_cu_conf():
    """Configuration dictionary with default values for cuda engine

    Returns
    -------
    output : dict
        Returns a dictionary of base cuda conf settings.
    """

    conf = default_base_conf()
    for (key, val) in default_base_cu_opts().items():
        conf[key] = val['default']
    return conf


def app_conf_paths(app_names):
    """Return list of configuration file paths for app_names

    Parameters
    ----------
    app_names : list of str
        list of application names.

    Returns
    -------
    output : list of str
        Returns a list of configuration paths.
    """

    local_apps = ['%s.cfg' % app for app in app_names]
    global_recon = os.path.expanduser(os.path.join('~', '.cphcttoolbox'
            , 'default.cfg'))
    global_apps = [os.path.expanduser(os.path.join('~', '.cphcttoolbox'
                   , app)) for app in local_apps]
    return [global_recon] + global_apps + local_apps


class ParseError(StandardError):

    """Simple dummy wrapper used to forward all parsing errors"""

    pass


def parse_command(argv, opt_handlers, conf):
    """Parse command line options and insert parsed values in conf

    Parameters
    ----------
    argv : list of str
        The list of call arguments.
    opt_handlers : dict
        A dictionary of application option specs used for parsing and doc.
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : dict
        Returns an updated configuration dictionary with values parsed from
        command line arguments inserted.
    """

    short_opts = ''
    long_opts = []
    handlers = {}

    # We need to delay help handler until this point where we know handlers
    # Put other options which only make sense for command line use here, too

    cli_opts = {'help': {
        'long': 'help',
        'short': 'H',
        'args': None,
        'handler': lambda val: help_exit(argv, opt_handlers, conf,
                val),
        'default': None,
        'description': 'Show this help',
        }, 'version': {
        'long': 'version',
        'short': 'V',
        'args': None,
        'handler': version_exit,
        'default': None,
        'description': 'Show version',
        }}

    opt_handlers.update(cli_opts)
    for (key, val) in opt_handlers.items():

        # add key to handler dictionary for later use

        val['conf'] = key

        # automatically insert underscore alias for long opts with hyphen

        val['alias'] = None
        if val.get('long', None):
            val['alias'] = val['long'].replace('-', '_')

        short_val = '%(short)s' % val
        long_val = '%(long)s' % val
        alias_val = '%(alias)s' % val
        if val.get('args', None):
            short_val += ':'
            long_val += '='
            alias_val += '='
        if val['short']:

            # Prevent illegal (duplicate or multi-letter) short options

            if len(val['short']) > 1:
                raise ParseError('illegal short option "%(short)s" (1 letter!)'
                                  % val)
            elif val['short'] in short_opts:
                raise ParseError('duplicate short option "%(short)s"'
                                 % val)
            handlers['-%(short)s' % val] = val
            short_opts += short_val
        if val['long']:
            handlers['--%(long)s' % val] = val
            long_opts.append(long_val)
        if val['alias']:
            handlers['--%(alias)s' % val] = val
            long_opts.append(alias_val)

    try:
        (opts, args) = getopt.getopt(argv[1:], short_opts, long_opts)
    except getopt.GetoptError, err:
        raise ParseError('command line option parsing failed: %s'
                         % err.msg)

    for (key, val) in opts:
        if key in handlers:
            entry = handlers[key]
            if entry['handler'] is not None:
                try:
                    conf[entry['conf']] = entry['handler'](val)
                except Exception, err:
                    raise ParseError("invalid '%s' option: %s" % (key,
                            err))
        else:
            raise ParseError('unknown command line option:: %s' % key)

    # Pass non-option arguments as well

    conf['args'] = args
    return conf


def conf_expander(opt_handlers, conf):
    """Expand all conf values with an associated expander function

    Parameters
    ----------
    opt_handlers : dict
        A dictionary of application option specs used for parsing and doc.
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : dict
        Returns configuration with values expanded.
    """

    for (key, val) in opt_handlers.items():
        if val.get('expander', None):
            val['expander'](conf, key)
    return conf


class FlatConfigParser(ConfigParser.SafeConfigParser):

    """Custom ConfigParser to parse configuration files without sections
    Inspired by suggestion from:
    http://mail.python.org/pipermail/python-dev/2002-November/029987.html
    """

    def read(self, filenames):
        """Read and parse a filename or a list of filenames without sections.
        This is a modified version of the default method shipped in python-2.7.

        Files that cannot be opened are silently ignored; this is
        designed so that you can specify a list of potential
        configuration file locations (e.g. current directory, user's
        home directory, systemwide directory), and all existing
        configuration files in the list will be read.  A single
        filename may also be given.

        Return list of successfully read files.

        Parameters
        ----------
        filenames : list of str
            list of configuration file names.

        Returns
        -------
        output : list of str
            Returns a list of paths for the configuration files that were read.
        """

        if isinstance(filenames, basestring):
            filenames = [filenames]
        read_ok = []
        for filename in filenames:
            try:
                raw = open(filename).read()
            except IOError:
                continue
            else:
                fake = StringIO.StringIO('[ALL]\n' + raw)
                self.readfp(fake, filename)
            read_ok.append(filename)
        return read_ok

    def check_missing(self, conf_paths):
        """Check that all paths in *conf_paths* are acual files.

        Parameters
        ----------
        conf_paths : list of str
            A list of configuration file paths to check.

        Raises
        ------
        ParseError
            If *mandatory_paths* contains any non-existing files
        """

        missing = []
        for path in conf_paths:
            if not os.path.isfile(path):
                missing.append(path)
        if missing:
            raise ParseError('no such configuration file(s): %s'
                             % ', '.join(missing))


def parse_config(
    optional_paths,
    mandatory_paths,
    opt_handlers,
    conf,
    ):
    """Parse configuration files in optional_paths and mandatory_paths lists
    and insert parsed values in conf. All paths in mandatory_paths must exist.

    Parameters
    ----------
    conf_paths : list of str
        list of configuration file names.
    opt_handlers : dict
        A dictionary of application option specs used for parsing and doc.
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : dict
        Returns an updated configuration dictionary with parsed configuration
        values inserted.

    Raises
    ------
    ParseError
        If *mandatory_paths* contains any non-existing files
    """

    conf_paths = mandatory_paths + optional_paths
    parser = FlatConfigParser()
    parser.read(optional_paths)
    parser.check_missing(mandatory_paths)
    parser.read(mandatory_paths)
    if not parser.has_section('ALL'):
        return conf
    for (key, val) in parser.items('ALL'):
        entry = opt_handlers.get(key, None)
        if not entry:
            raise ParseError('unknown configuration option in %s:: %s'
                             % (', '.join(conf_paths), key))
        if entry['handler'] is not None:
            try:
                conf[key] = entry['handler'](val)
            except Exception, err:
                raise ParseError("invalid '%s' configuration value: %s"
                                 % (key, err))
    return conf


def parse_setup(
    argv,
    app_names,
    opt_handlers,
    conf,
    ):
    """Combined configuration file and command line option parser to simplify
    the steps of:
     * reading extra conf file paths from command line
     * parsing all conf files
     * updating the provided default conf with the parsed conf file values
     * parsing all command line options
     * overriding conf with command line option values
     * expand all relative paths to use actual working_directory prefix

    Parameters
    ----------
    argv : list of str
        The list of call arguments.
    app_names : list of str
        The list of application names.
    opt_handlers : dict
        A dictionary of application option specs used for parsing and doc.
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : dict
        Returns an updated configuration dictionary with parsed configuration
        values inserted.
    """

    # Only extract additional conf files from comand line

    dummy = parse_command(argv, opt_handlers, conf)
    optional_files = app_conf_paths(app_names)
    required_files = dummy['args']
    conf = parse_config(optional_files, required_files, opt_handlers,
                        conf)
    conf = parse_command(argv, opt_handlers, conf)
    conf = conf_expander(opt_handlers, conf)
    return conf


