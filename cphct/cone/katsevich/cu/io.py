#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - cuda specific input/ouput helpers
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

"""CUDA specific input/output helper functions"""

from cphct.io import expand_path
from cphct.cone.cu.io import fill_cone_cu_conf
from cphct.cone.katsevich.npycore.io import fill_katsevich_npycore_conf
from cphct.npycore import sqrt


def __get_smallest_scaler(value):
    """Returns a the smallest integer value that value can be downscaled with.
    All integer values from 2 up to the squareroot of value are tested in
    turn. If none of them are divisors of value it returns 1.

    Parameters
    ----------
    value : int
        Value to find smallest divisor greater than 1 for

    Returns
    -------
    output : int
        Returns the smallest possible scale integer for value.
    """

    for i in xrange(2, int(sqrt(value))+1):
        if value % i == 0:
            return i
    return 1

def fill_katsevich_cu_conf(conf):
    """Remaining configuration after handling command line options.
    Casts all floating point results using float data type from conf.
    This version is specifically for the cuda engine.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary filled with cuda specific settings.
    """

    fill_cone_cu_conf(conf)
    fill_katsevich_npycore_conf(conf)
    
    # Set up additional vars based on final conf
    
    fdt = conf['data_type']
    conf['load_gpu_kernels_path'] = expand_path(
        conf['cphcttoolbox_base'], 'cphct/cone/katsevich/cu/kernels/base.cu')
                                                
    if conf['precision'] != 'float32':
        raise ValueError('cukatsevich only supports \'float32\'')

    proj_bytes = fdt(0.0).nbytes * conf['detector_rows'] * \
               conf['detector_columns']

    # Override default filter chunk size to fit in GPU mem
    # We use 5 filtering buffers

    size = 5 * proj_bytes
    keep_going = True
    while keep_going:
        keep_going = False
        if size * conf['filter_out_projs'] > conf['gpu_target_filter_memory']:
            scaler = __get_smallest_scaler(conf['filter_out_projs'])
            if scaler > 1:
                conf['filter_out_projs'] /= scaler
                keep_going = True

    conf['filter_in_projs'] = conf['filter_out_projs'] \
        + conf['extra_filter_projs']

    conf['backproject_in_projs'] = conf['chunk_projs']

    # Override default backproject projection chunk size to fit in GPU mem

    size = 1 * proj_bytes
    keep_going = True
    while keep_going:
        keep_going = False
        if size * conf['backproject_in_projs'] > conf['gpu_target_input_memory']:
            scaler = __get_smallest_scaler(conf['backproject_in_projs'])
            if scaler > 1:
                conf['backproject_in_projs'] /= scaler
                keep_going = True

    return conf
