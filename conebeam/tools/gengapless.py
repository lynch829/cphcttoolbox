#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# gengapless - cone beam projection gapless resampler
# Copyright (C) 2012  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Resample cone beam projections with pixel gaps to gapless ones"""

import sys

from cphct.cone.conf import str_value, gap_pair_values, int_value, \
    default_cone_npy_conf, default_cone_npy_opts, enable_conf_option, \
    engine_opts, engine_conf, parse_setup, ParseError
from cphct.cone.npy.io import fill_cone_npy_conf
from cphct.cone.npycore.utils import general_tool_helper, default_tool_opts, \
    default_init_tool, default_exit_tool as exit_tool, engine_app_aliases, \
    resample_gapless

app_names = [__file__]


def tool_opts():
    """Options for this tool

    Returns
    -------
    output : dict
        Returns a dictionary of processed options helper dictionaries.
    """

    opts = default_tool_opts()
    opts.update({
        'detector_row_gaps': {
            'long': 'detector-row-gaps',
            'short': None,
            'args': str,
            'handler': gap_pair_values,
            'default': [],
            'description': 'Row gap positions measured in floating point ' \
                + 'pixel widths',
            },
        'detector_column_gaps': {
            'long': 'detector-column-gaps',
            'short': None,
            'args': str,
            'handler': gap_pair_values,
            'default': [],
            'description': 'Column gap positions measured in floating point ' \
                + 'pixel heights',
            },
        'detector_resample_rows': {
            'long': 'detector-resample-rows',
            'short': None,
            'args': str,
            'handler': int_value,
            'default': -1,
            'description': 'Number of pixel rows in resampled projections',
            },
        'detector_resample_columns': {
            'long': 'detector-resample-columns',
            'short': None,
            'args': str,
            'handler': int_value,
            'default': -1,
            'description': 'Number of pixel columns in resampled projections',
            },
        'detector_gap_interpolation': {
            'long': 'detector-gap-interpolation',
            'short': None,
            'args': str,
            'handler': str_value,
            'default': 'linear',
            'allowed': ['point', 'linear'],
            'description': 'Interpolation method in gapless pixel resampling',
            },
        })
    return opts

def tool_conf():
    """For enabling option of same name in conf

    Returns
    -------
    output : dict
        Returns a configuration dictionary where the related options are set.
    """

    return enable_conf_option(tool_opts())

def init_tool(conf):
    """Init tool helpers

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.

    Returns
    -------
    output : dict
        Returns a dictionary of tool helper variables.
    """

    state = default_init_tool(conf)

    # Init output size - default resample to detector size if not set

    if conf['detector_resample_rows'] > 0:
        state['output_rows'] = conf['detector_resample_rows']
    else:
        state['output_rows'] = conf['detector_rows']
    if conf['detector_resample_columns'] > 0:
        state['output_columns'] = conf['detector_resample_columns']
    else:
        state['output_columns'] = conf['detector_columns']
    return state

def core_tool(projs, conf, tool_state):
    """Core tool function: takes a chunk of projections and returns
    the same processed chunk based on conf and state settings.

    Parameters
    ----------
    projs : array
        An array of stacked projections to process.
    conf : dict
        A dictionary of configuration options.
    tool_state : dict
        A dictionary of tool helper variables.
    """

    return resample_gapless(projs, conf, conf['detector_row_gaps'], 
                            conf['detector_column_gaps'], 
                            tool_state['output_rows'],
                            tool_state['output_columns'],
                            conf['detector_gap_interpolation'])

def main(conf):
    """Resample to gapless projections

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    """

    fill_cone_npy_conf(conf)
    general_tool_helper(app_names, conf, init_tool, core_tool, exit_tool)


if __name__ == '__main__':
    cone_cfg = default_cone_npy_conf()

    # Override default value for preprocessed settings and engine

    cone_cfg.update(tool_conf())
    cone_cfg.update(engine_conf())

    cone_opts = default_cone_npy_opts()

    # Override default no-op action for preprocessed settings and engine

    cone_opts.update(tool_opts())
    cone_opts.update(engine_opts())
    try:
        cone_cfg = parse_setup(sys.argv, app_names, cone_opts, cone_cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(1)
    app_names += engine_app_aliases(app_names, cone_cfg)
    main(cone_cfg)
