#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# log - logging helpers
# Copyright (C) 2011  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Logging helper functions"""

import logging
import os

allowed_log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
    }
default_level = logging.INFO
default_format = '%(asctime)s %(levelname)s %(message)s'


def setup_log(path, log_level=None, log_format=None):
    """Set up logging to use file or stdout if no path is given

    Parameters
    ----------
    path : str
        log file path or empty to use stdout.
    log_level : str
        A string indicating log verbosity. Used as key lookup in
        allowed_log_levels dictionary.
    log_format : str
        A format string to use for log entries.

    Returns
    -------
    output : object
        Returns a configured logging object.
    """

    # Translate level string to real level

    if log_level in allowed_log_levels.keys():
        log_level = allowed_log_levels[log_level]

    # Fall back to default_level if invalid

    if not log_level in allowed_log_levels.values():
        log_level = default_level
    if log_format is None:
        log_format = default_format
    if path:

        # Make sure parent dir exists

        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass
        logging.basicConfig(filename=path, level=log_level,
                            format=log_format)
    else:
        logging.basicConfig(level=log_level, format=log_format)
    return logging

def log_scan_geometry(conf, opts):
    """Log scanner geometry in a pretty format

    Parameters
    ----------
    conf : dict
        Configuration dictionary with all geometry settings.
    opts : dict
        options dictionary.
    """
    
    geometry_fields = {}
    info_fields = ["detector_shape", "detector_rows", "detector_columns",
                   "detector_pixel_height", "detector_pixel_width",
                   "detector_height", "detector_width",
                   "detector_distance", "source_distance",
                   "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
                   "x_voxels", "y_voxels", "z_voxels", 
                   "projs_per_turn", "progress_per_turn", "chunk_size",
                   ]
    debug_fields = ["total_turns", "detector_row_offset",
                    "detector_column_offset", "detector_rebin_rows"]
    for field in info_fields:
        if field in conf:
            geometry_fields[field] = {"title": opts[field]["description"],
                                      "log_func": logging.info}
    for field in debug_fields:
        if field in conf:
            geometry_fields[field] = {"title": opts[field]["description"],
                                      "log_func": logging.debug}
    logging.info("****************** Scanner geometry ******************")
    for field in info_fields + debug_fields: 
        if not field in geometry_fields:
            continue
        specs = geometry_fields[field]
        specs["log_func"]("%s: %s" % (specs["title"], conf[field]))
    logging.info("******************************************************")
