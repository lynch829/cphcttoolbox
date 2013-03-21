#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# init - numpy specific katsevich reconstruction kernels
# Copyright (C) 2011-2012  The CT-Toolbox Project lead by Brian Vinter
#
# This file is part of CT-Toolbox.
#
# CT-Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# CT-Toolbox is distributed in the hope that it will be useful,
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

"""Spiral cone beam CT kernels using the Katsevich algorithm"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# Functions available through cphct.cone.katsevich.npycore.kernels

from cphct.cone.katsevich.npycore.kernels.base import flat_diff_chunk_vector, \
    flat_fwd_rebin_chunk_orig, flat_fwd_rebin_chunk_single, \
    flat_fwd_rebin_chunk_vector, flat_conv_chunk_vector, \
    flat_rev_rebin_chunk_single, curved_diff_chunk_vector, \
    curved_fwd_rebin_chunk_vector, curved_conv_chunk_vector, \
    curved_rev_rebin_chunk_single, flat_backproject_chunk, \
    curved_backproject_chunk, filter_chunk, backproject_chunk
from cphct.cone.katsevich.npycore.kernels.initialize import init_recon

# All sub modules to load in case of 'from X import *'

__all__ = []
