#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - cone specific input/ouput helpers
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

"""Cone specific input/output helper functions"""


def fill_cone_conf(conf):
    """Remaining configuration after handling command line options.
    Expand all relative paths to absolute paths

    Parameters
    ----------
    conf : dict
        Configuration dictionary to be filled.

    Returns
    -------
    output : dict
        Filled configuration dictionay where all the base configuration
        options are set.
    """

    # Set up additional vars based on final conf

    conf['app_state']['chunk'] = {}
    if conf['chunk_size'] < 0:
        conf['chunk_size'] = conf['z_voxels']

    return conf
