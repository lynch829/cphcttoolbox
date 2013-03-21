#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - Back end functions shared by plugins and tools.
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

"""Cph CT Toolbox back end functions shared by plugins and tools.
We separate I/O from the actual handlers so that they can be used inside apps
and in separate tools scripts."""

import os

def transfer_missing_metadata(src, dst, conf):
    """Transfer missing meta data files in src to dst filtering file paths
    accordingly.
    Returns names of files actually found and transferred.

    Parameters
    ----------
    src : str
        path to src directory.
    dst : str
        path to dst directory.
    conf : dict
        dictionary with configuration values.

    Returns
    -------
    output : list of str
        Returns list of transferred metadata file names.
    """

    xfer_files = []
    meta_data_files = conf.get('meta_data_files', ['scene.txt',
                               'angles.txt'])
    meta_data_files.append(os.path.basename(conf.get('load_scene_path',
                           '')))
    for metadata_file in meta_data_files:
        src_path = os.path.join(src, metadata_file)
        dst_path = os.path.join(dst, metadata_file)
        if os.path.isfile(src_path) and not os.path.exists(dst_path):
            src_fd = open(src_path, 'r')
            dst_fd = open(dst_path, 'w')
            for line in src_fd:
                dst_fd.write(line.replace(src, dst))
            src_fd.close()
            dst_fd.close()
            xfer_files.append(metadata_file)
    return xfer_files


