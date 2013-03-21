#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# gennormalized - cone beam projection value normalizer
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

"""Generate value normalized projs from cone beam projection files"""

import sys

from cphct.cone.conf import float_value, default_cone_npy_conf, \
    default_cone_npy_opts, enable_conf_option, engine_opts, engine_conf, \
    parse_setup, ParseError
from cphct.cone.npy.io import fill_cone_npy_conf
from cphct.cone.npycore.utils import inline_tool_helper, default_tool_opts, \
    default_init_tool as init_tool, default_exit_tool as exit_tool, \
    engine_app_aliases, normalize_array

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
        'normalize_min': {
            'long': 'normalize-min',
            'short': None,
            'args': str,
            'handler': float_value,
            'default': None,
            'description': 'Clip all projection values to at least this value',
            },
        'normalize_max': {
            'long': 'normalize-max',
            'short': None,
            'args': str,
            'handler': float_value,
            'default': None,
            'description': 'Clip all projection values to at most this value',
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

    (normalize_min, normalize_max) = (conf['normalize_min'], conf['normalize_max'])
    normalize_array(projs, normalize_min, normalize_max)
    return projs


def main(conf):
    """Generate normalized projs

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    """

    fill_cone_npy_conf(conf)
    inline_tool_helper(app_names, conf, init_tool, core_tool, exit_tool)


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
