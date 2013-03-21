#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - fdk specific input/ouput helpers
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

"""FDK specific input/output helper functions"""

from cphct.conf import allowed_values, allowed_scanning_paths

def fill_fdk_conf(conf):
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

    Raises
    ------
    ValueError
       If chunk_size is set and isn't a divisor of z_voxs
    """

    # Set up additional vars based on final conf

    if conf['total_turns'] < 0:
        conf['total_turns'] = 1

    conf['total_projs'] = int(conf['total_turns']
                              * conf['projs_per_turn'])

    # TODO: do we actually want to enforce that chunk limitation?

    if conf['z_voxels'] % conf['chunk_size'] != 0:
        msg = 'Invalid chunk_size: \'%s\',' % conf['chunk_size']
        msg = '%s must be a divisor of z_voxels: \'%s\'' % (msg,
                conf['z_voxels'])
        raise ValueError(msg)

    conf['scanning_path'] = allowed_values('step',
            allowed_scanning_paths)

    return conf


