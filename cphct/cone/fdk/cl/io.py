#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - OpenCL specific input/ouput helpers
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

"""OpenCL specific input/output helper functions"""

from cphct.io import expand_path
from cphct.cone.cl.io import fill_cone_cl_conf
from cphct.cone.fdk.gpu.io import fill_fdk_gpu_conf


def fill_fdk_cl_conf(conf):
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
        
    Raises
    ------
    ValueError:
        If precision != float32
    """

    fill_cone_cl_conf(conf)
    fill_fdk_gpu_conf(conf)

    # Set Up Additional vars based on final conf
    
    conf['load_gpu_kernels_path'] = expand_path(
        conf['cphcttoolbox_base'], 'cphct/cone/fdk/cl/kernels/base.cl')

    if conf['precision'] != 'float32':
        raise ValueError('clfdk only supports \'float32\'')
        
    return conf
