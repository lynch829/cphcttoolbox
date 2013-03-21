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

from cphct.npycore import pi, cos, tan, ceil
from cphct.npycore.io import get_npy_size
from cphct.cone.katsevich.io import fill_katsevich_conf


def fill_katsevich_npycore_conf(conf):
    """Remaining configuration after handling command line options.
    Casts all floating point results using float data type from conf.
    This version is for the shared numpy core.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary filled with numpy core settings.
    """

    fill_katsevich_conf(conf)

    fdt = conf['data_type']

    # Make sure all float values get the right precision before we continue

    for (key, val) in conf.items():
        if isinstance(val, float):
            conf[key] = fdt(val)

    # Helix path

    conf['progress_per_radian'] = fdt(conf['progress_per_turn'] / (2
            * pi))
    conf['s_min'] = fdt(-pi + conf['z_min'] / conf['progress_per_radian'
                        ])
    conf['s_max'] = fdt(pi + conf['z_max'] / conf['progress_per_radian'
                        ])
    conf['s_len'] = fdt(conf['s_max'] - conf['s_min'])
    conf['delta_s'] = fdt(2 * pi / conf['projs_per_turn'])

    # rebinning coords in [-pi / 2 - half_fan_angle : pi / 2 + half_fan_angle]
    # pixel center at end points, so detector_rebin_rows minus one half in
    # each end

    conf['detector_rebin_rows_height'] = fdt((pi + 2
            * conf['half_fan_angle']) / (conf['detector_rebin_rows']
            - 1))

    # convolution helpers

    conf['kernel_radius'] = conf['detector_columns'] - 1
    conf['kernel_width'] = 1 + 2 * conf['kernel_radius']

    # Auto select optimal detector size if not provided
    
    if conf['detector_pixel_width'] <= 0.0 \
        or conf['detector_pixel_height'] <= 0.0 or conf['detector_width'
            ] <= 0.0 or conf['detector_height'] <= 0.0:

        # Make sure detector is big enough to avoid truncation along the
        # axes (Tam-Danielsson window).
        # Please refer to 4.2/5.2 in Noo, Pack and Heuscher paper for the
        # details in relation to flat and curved detectors:
        # "Exact helical reconstruction using native cone-beam geometries"
        # detector_half_width is the distance to the outermost pixel center
        #
        # Avoid rounding issues from actually resulting in too small a
        # detector by forcing the window to fit in one fewer rows and columns

        if conf['detector_shape'] == 'flat':
            conf['detector_half_width'] = fdt(conf['scan_diameter']
                    * tan(conf['half_fan_angle']))

            conf['detector_pixel_width'] = fdt(2.0
                    * conf['detector_half_width']
                    / max(conf['detector_columns'] - 2, 1))
            conf['detector_pixel_height'] = \
                fdt((conf['detector_half_width'] ** 2
                    + conf['scan_diameter'] ** 2) * (pi / 2
                    + conf['half_fan_angle']) * conf['progress_per_turn'
                    ] / (pi * max(conf['detector_rows'] - 2, 1)
                    * conf['scan_radius'] * conf['scan_diameter']))
        elif conf['detector_shape'] == 'curved':
            conf['detector_pixel_width'] = fdt(2.0
                    * conf['scan_diameter']
                    * conf['half_fan_angle']
                    / max(conf['detector_columns'] - 2, 1))
            conf['detector_pixel_height'] = fdt(conf['scan_diameter']
                    * conf['progress_per_turn'] * (pi / 2
                    + conf['half_fan_angle']) / (max(conf['detector_rows']
                    - 2, 1) * pi * conf['scan_radius']
                    * cos(conf['half_fan_angle'])))

    # Now map measured curve length to polar angle span if curved

    if conf['detector_shape'] == 'flat':
        conf['detector_pixel_span'] = conf['detector_pixel_width']
    else:
        conf['detector_pixel_span'] = fdt(
            conf['detector_pixel_width'] / conf['scan_diameter'])            

    # Printable detector dimensions in cm

    conf['detector_pixel_size'] = (conf['detector_pixel_height'],
                                   conf['detector_pixel_width'])
    conf['detector_height'] = fdt(conf['detector_pixel_size'][0]
                                  * conf['detector_rows'])
    conf['detector_width'] = fdt(conf['detector_pixel_size'][1]
                                 * conf['detector_columns'])
    conf['detector_half_width'] = fdt(0.5 * conf['detector_width'])
    conf['detector_size'] = (conf['detector_height'],
                             conf['detector_width'])

    # Calculate core scan turns from z length and progress if not set.
    # Please note that this calculation doesn't include overscan.

    if conf['total_turns'] < 0:
        conf['total_turns'] = int(ceil(conf['z_len']
                                  / conf['progress_per_turn']))

    # Always add two half rotations of overscan - not all projections are used

    conf['total_turns'] += 1

    conf['total_projs'] = int(conf['total_turns']
                              * conf['projs_per_turn'])

    # Map projections to each chunk

    overscan_projs = conf['overscan_projs'] = int(conf['projs_per_turn'
            ] / 2)
    base_projs = conf['total_projs'] - 2 * overscan_projs
    end_first_chunk_offset = (conf['chunk_size'] - 1.0) \
        / conf['z_voxels']
    begin_next_chunk_offset = 1.0 * conf['chunk_size'] / conf['z_voxels'
            ]
    conf['chunk_projs_offset'] = int(begin_next_chunk_offset
            * base_projs)
    conf['chunk_projs'] = int(end_first_chunk_offset * base_projs) + 2 \
        * overscan_projs + 1

    # Set up additional vars based on final conf

    # Handle one turn at a time - trade off between streaming and transfers

    conf['filter_out_projs'] = conf['projs_per_turn']

    # One additional projection needed for filtering

    conf['extra_filter_projs'] = 1
    conf['filter_in_projs'] = conf['filter_out_projs'] \
        + conf['extra_filter_projs']

    return conf
