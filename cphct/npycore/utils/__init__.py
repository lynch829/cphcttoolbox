#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# __init__ - numpy back end functions shared by plugin and tools
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

"""Cph CT Toolbox numpy back end functions shared by plugins and tools.
We separate I/O from the actual handlers so that they can be used inside apps
and in separate tools scripts."""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Functions available through cphct.npycore.utils

from cphct.npycore.utils.base import prepare_output, normalize_array, \
    clip_array, dump_array, square_array, flux_to_proj, \
    checksum_matrix, log_checksum, check_norm, interpolate_proj_pixels, \
    supported_proj_filters, generate_proj_filter, hounsfield_scale

# All sub modules to load in case of 'from X import *'

__all__ = []
