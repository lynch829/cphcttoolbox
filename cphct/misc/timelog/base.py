#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# timelog - time logging module shared by all modules.
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

"""Cph CT Toolbox time logging module shared by all modules."""

import time

allowed_timelogs = ['default', 'verbose']

__start_time = {}


def __barrier(conf):
    """Issues barrier based on conf settings
    
    Parameters
    ----------
    conf : dict
       Dictionary with configuration values.
       
    Returns
    -------
    output : dict
       Dictionary with configuration values. 
    """

    if 'context' in conf['gpu']:
        for gpu_id in conf['gpu']['context']:
            conf['gpu']['context'][gpu_id].synchronize()

    return conf


def init(conf, default, verbose):
    """Initialized timelog
    
    Parameters
    ----------
    conf : dict
       Dictionary with configuration values.
    default : list
       List of default logging entries
    vebose: list
       List of verbose logging entries
       
    Returns
    -------
    output : dict
       Dictionary updated with a timelog
    """

    conf['timelog_dict'] = {}
    conf['timelog_dict']['default'] = {}
    conf['timelog_dict']['verbose'] = {}

    for entry in default:
        conf['timelog_dict']['default'][entry] = 0.0

    for entry in verbose:
        conf['timelog_dict']['verbose'][entry] = 0.0

    return conf


def get(conf, mode, desc):
    """Get timelog entry
    
    Parameters
    ----------
    conf : dict
       Dictionary with configuration values.
    mode : str
       Log mode entry 'default' or 'verbose'
    desc : str
       Description of time log event
       
    Returns
    -------
    output : float
       Logged time value

    """

    return conf['timelog_dict'][mode].get(desc)


def set(
    conf,
    mode,
    desc,
    start_time=None,
    barrier=False,
    ):
    """Set execution start time
        
    Parameters
    ----------
    conf : dict
       Dictionary with configuration values.
    mode : str
       Log mode entry 'default' or 'verbose'
    desc : str
       Description of time log event
    start_time : float, optional
        Log entry start time
    barrier : bool, optional
       If *True* a barrier is issued before setting start time 
       
    Returns
    -------
    output : float
       Start time in seconds
    """

    if mode == 'default' or mode == 'verbose' and conf['timelog'] \
        == 'verbose':

        if barrier:
            __barrier(conf)

        if start_time is not None:
            __start_time[desc] = start_time
        else:
            __start_time[desc] = time.time()


def log(
    conf,
    mode,
    desc,
    start_time=None,
    end_time=None,
    barrier=False,
    ):
    """Log execution time
    
    Parameters
    ----------
    conf : dict
       dictionary with configuration values.
    mode : str
       Log mode entry 'default' or 'verbose'
    desc : str
       Description of time log event
    start_time : float, optional
       Log entry start time
    end_time : float, optional
       Log entry end time
    barrier : bool, optional
       If *True* a barrier is issued before time measurement

    Returns
    -------
    output : float
       Returns time added to log
       
    """

    diff_time = 0.0

    if mode == 'default' or mode == 'verbose' and conf['timelog'] \
        == 'verbose':

        if barrier:
            __barrier(conf)

        if start_time is None:
            start_time = __start_time[desc]

        if end_time is None:
            end_time = time.time()

        diff_time = end_time - start_time

        conf['timelog_dict'][mode][desc] += end_time - start_time

    return diff_time


