#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - OpenCL specific input/ouput helpers
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

"""OpenCL specific input/output helper functions"""

from cphct.npycore.io import fill_base_npycore_conf
from cphct.io import engine_alloc, get_engine_data, get_engine_size, \
    get_engine_total_size, engine_free, engine_free_all


def fill_base_cl_conf(conf):
    """Remaining configuration after handling command line options.
    Casts all floating point results using float data type from conf.
    This version is specifically for the OpenCL engine.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary filled with OpenCL specific settings.
    """

    fill_base_npycore_conf(conf)

    # Make sure engine is set for e.g. plugins

    if not conf['engine']:
        conf['engine'] = 'opencl'

    # Initiate dict for OpenCL allocations and structures

    conf['cl_data'] = {}
    conf['compute'] = {}

    # Some options imply copy back to host

    if conf['checksum'] or conf['save_filtered_projs_data_path']:
        conf['gpu_projs_only'] = False

    return conf

def cl_alloc(
    conf,
    key,
    data,
    size,
    ):
    """Stores a {'data': data, 'size': size} entry with given key in
    conf['cl_data'] .
    Used to keep track of allocated OpenCL data and provide shared access.

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    key : str
        Variable name for *data*
    data : object
        Allocated data array
    size : int
        Size of allocated data array

    Returns
    -------
    output : dict
        Same configuration dictionary where the conf['cl_data'] dictionary is
        extended to map *key* to a new dictionary with *data* and *size*.

    Raises
    ------
    ValueError
        If *key* is *None* or
        if *key* is already in allocated set or
    """

    return engine_alloc(conf, 'cl_data', key, data, size)


def get_cl_data(conf, key):
    """Extracts OpenCL data for variable *key*

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    key : str
        Data entry dict key

    Returns
    -------
    output : object
        Data entry corresponding to *key* in conf['cl_data']
        *None* if *key* is not in conf['cl_data']
    """

    return get_engine_data(conf, 'cl_data', key)


def get_cl_size(conf, key):
    """Extracts OpenCL data size for variable *key*

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    key : str
        Data entry dict key

    Returns
    -------
    output : int
        Size entry corresponding to *key* in conf['cl_data']
        Zero if *key* is not in conf['cl_data']
    """

    return get_engine_size(conf, 'cl_data', key)


def get_cl_total_size(conf):
    """Extracts the total size of allocated OpenCL data

    Parameters
    ----------
    conf : dict
        Configuration dictionary

    Returns
    -------
    output : int
        Sum of all size entries in conf['cl_data']
    """

    return get_engine_total_size(conf, 'cl_data')


def cl_free(conf, key, ignore_missing=False):
    """Free OpenCL data entry *key* from conf['cl_data'].

    Does not explicitly free the data, but free happens automatically during
    garbage collection when no more references to the GPUArray or low-level
    mem_alloc exist.

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    key : str
        Variable to be freed
    ignore_missing : bool, optional
        If *True* the OpenCL data for variable *key* is freed if present.
        If *False* exceptions may be raised
        based on the value of *key* (see Raises),

    Returns
    -------
    output : dict
        Configuration dictionary

    Raises
    ------
    ValueError
        If *key* is None or
        if *key* isn not in allocated set
    """

    return engine_free(conf, 'cl_data', key, ignore_missing)


def cl_free_all(conf, garbage_collect=True):
    """Free all OpenCL data entries from conf['cl_data'].

    Unreachable allocations from both GPUArrays and low-level mem_alloc are
    automatically freed during garbage collect, so there's no need to
    explicitly free them as long as no other references are preserved.

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    garbage_collect : bool
        Flag to disable explicit garbage collection as last action

    Returns
    -------
    output : dict
        Configuration dictionary
    """

    return engine_free_all(conf, 'cl_data', garbage_collect)
