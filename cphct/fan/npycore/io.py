#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - NumPy core specific input/ouput helpers
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

"""NumPy core specific input/output helper functions"""

from cphct.fan.io import fill_fan_conf


def fill_fan_npycore_conf(conf):
    """Remaining configuration after handling command line options.
    Casts all floating point results using float data type from conf.
    This version is for the shared NumPy core.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary filled with NumPy core settings.
    """

    fill_fan_conf(conf)
    fdt = conf['data_type']

    # Make sure all float values get the right precision before we continue

    for (key, val) in conf.items():
        if isinstance(val, float):
            conf[key] = fdt(val)

    return conf
