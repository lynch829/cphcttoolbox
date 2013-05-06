#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - numpy core specific input/ouput helpers
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

"""Numpy core specific input/output helper functions"""

from cphct.cone.io import fill_cone_conf
from cphct.npycore import ceil


def fill_cone_npycore_conf(conf):
    """Remaining configuration after handling command line options.
    Casts all floating point results using float data type from conf.
    This version is for the shared numpy core.

    Parameters
    ----------
    conf : dict
        Configuration dictionary to be filled.

    Returns
    -------
    output : dict
        Returns configuration dictionary filled with numpy core settings.
    """

    fill_cone_conf(conf)

    fdt = conf['data_type']

    # Make sure all float values get the right precision before we continue

    for (key, val) in conf.items():
        if isinstance(val, float):
            conf[key] = fdt(val)

    conf['z_len'] = fdt(conf['z_max'] - conf['z_min'])

    # Important: by definition x, y, z voxels start and end with *center* at
    # *_min and *_max values.
    # I.e. the boundary voxels occuppy only *half* a voxel inside the range
    # [*_min, *_max] leaving delta_* slightly bigger than *_len/*_voxs .
    # (one might argue that *_min is really min plus delta_*/2, but anyway)

    conf['delta_z'] = fdt(conf['z_len'] / max(conf['z_voxels'] - 1, 1))

    # Calculate generic pixel size for case without auto sizing

    if conf['detector_pixel_height'] > 0.0:
        conf['detector_height'] = fdt(1.0 * conf['detector_pixel_height'
                ] * conf['detector_rows'])
    elif conf['detector_height'] > 0.0:

        conf['detector_pixel_height'] = fdt(1.0 * conf['detector_height'
                ] / conf['detector_rows'])

    # Chunk defaults

    conf['chunk_count'] = int(ceil(1.0 * conf['z_voxels']
                              / conf['chunk_size']))
    if not conf['chunk_range']:
        conf['chunk_range'] = [0, conf['chunk_count'] - 1]

    # Sanitize range list and make an iterator of actual chunk indices

    conf['chunk_range'] = (conf['chunk_range'])[:2]
    conf['chunk_range'][-1] = min(conf['chunk_range'][-1],
                                  conf['chunk_count'] - 1)
    conf['chunks_enabled'] = xrange(conf['chunk_range'][0],
                                    conf['chunk_range'][1] + 1)

    # We use the (possibly offset) detector pixel shift in helpers and in
    # reconstruction. The '-1' is due to the use of pixel centers.

    conf['detector_row_shift'] = -conf['detector_row_offset'] + \
                                 0.5 * (conf['detector_rows'] - 1)
    conf['detector_column_shift'] = -conf['detector_column_offset'] + \
                                    0.5 * (conf['detector_columns'] - 1)

    return conf
