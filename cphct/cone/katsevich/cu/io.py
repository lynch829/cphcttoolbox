#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - cuda specific input/ouput helpers
# Copyright (C) 2011-2012  The Cph CT Toolbox Project lead by Brian Vinter
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

    # Override default filter chunk size to fit in GPU mem
    # We use 5 filtering buffers

    size = 5 * fdt(0.0).nbytes * conf['detector_rows'] * \
               conf['detector_columns']
    keep_going = True
    while keep_going:
        keep_going = False
        if size * conf['filter_out_projs'] > conf['gpu_target_filter_memory']:
            for scale in (2, 3, 5):
                if conf['filter_out_projs'] % scale == 0:
                    conf['filter_out_projs'] /= scale
                    keep_going = True
                    break
    #print "DEBUG: using buf size %d (%d)" % (size * conf['filter_out_projs'])
    #print "DEBUG: using filter out projs %d" % (conf['filter_out_projs'])

    conf['filter_in_projs'] = conf['filter_out_projs'] \
        + conf['extra_filter_projs']

    return conf
