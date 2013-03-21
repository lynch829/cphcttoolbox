#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# genslices - cone beam projection sub slice generator
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

"""Generate sub slices from full cone beam projection files"""

import sys

from cphct.cone.conf import float_value, default_cone_npy_conf, \
    default_cone_npy_opts, enable_conf_option, engine_opts, engine_conf, \
    parse_setup, ParseError
from cphct.cone.npy.io import fill_cone_npy_conf
from cphct.cone.npycore.utils import general_tool_helper, default_tool_opts, \
    default_init_tool, default_exit_tool as exit_tool, engine_app_aliases, \
    extract_slices

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
        'first_row': {
            'long': 'first-row',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 0,
            'description': 'Zero-based floating point index of first row to ' \
                + 'include in slice',
            },
        'last_row': {
            'long': 'last-row',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1,
            'description': 'Zero-based floating point index of last row to ' \
                + 'include in slice',
            },
        'first_column': {
            'long': 'first-column',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': 0,
            'description': 'Zero-based floating point index of first column ' \
                + 'to include in slice',
            },
        'last_column': {
            'long': 'last-column',
            'short': None,
            'args': float,
            'handler': float_value,
            'default': -1,
            'description': 'Zero-based floating point index of last column ' \
                + 'to include in slice',
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

    # Include last item

    state['output_rows'] = 1 + conf['last_row'] - conf['first_row']
    state['output_columns'] = 1 + conf['last_column'] - conf['first_column']
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

    conf['slice_rows'] = tool_state['output_rows']
    conf['slice_columns'] = tool_state['output_columns']
    return extract_slices(projs, conf)

def main(conf):
    """Generate sub slices

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
