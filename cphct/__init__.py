#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# __init__ - shared lib module init
# Copyright (C) 2011-2014  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Cph CT Toolbox shared library module initializer"""

__dummy = \
    '''This dummy right after the module doc string prevents PythonTidy
from incorrectly moving following comments above module doc string'''

# All sub modules to load in case of 'from X import *'

__all__ = [
    'conf',
    'io',
    'log',
    'plugins',
    'utils',
    'cone',
    'fan',
    'cu',
    'npy',
    'npycore',
    'ocl',
    ]

# Collect all package information here for easy use from scripts and helpers

package_name = 'Cph CT Toolbox'
short_name = 'cphcttoolbox'

# IMPORTANT: Please keep version in sync with doc-src/README.t2t

version_tuple = (1, 0, 4)
version_suffix = ''
version_string = '.'.join([str(i) for i in version_tuple]) + version_suffix
package_version = '%s %s' % (package_name, version_string)
project_team = 'The Cph CT Toolbox project lead by Brian Vinter'
project_email = 'brian DOT vinter AT gmail DOT com'
maintainer_team = 'The Cph CT Toolbox maintainers'
maintainer_email = 'jonas DOT bardino AT gmail DOT com'
project_url = 'http://code.google.com/p/cphcttoolbox/'
download_url = 'http://pypi.python.org/pypi/cphcttoolbox/'
license_name = 'GNU GPL v2'
short_desc = \
    'Cph CT Toolbox is a selection of Computed Tomography tools'
long_desc = \
    """Copenhagen Computed Tomography Toolbox is a collection of
applications and libraries for flexible and efficient CT reconstruction. The
toolbox generally take a set of projections (X-ray intensity measurements) and
filter and back project them in order to recreate the image or volume that the
projections represent.
The project includes both mostly informative CPU implementations and efficient
GPU implementations.
"""
project_class = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    ]
project_keywords = [
    'computed',
    'tomography',
    'science',
    'research',
    'cpu',
    'gpu',
    ]

# We can't really do anything useful without at least numpy

full_requires = [('numpy', '>=1.3')]
versioned_requires = [''.join(i) for i in full_requires]
project_requires = [i[0] for i in full_requires]

# Optional packages required for additional functionality (for extras_require)

project_extras = {'CUDA': ['pycuda'], 'OpenCL': ['pyopencl'],
                  'PIL': ['imaging']}
package_provides = short_name
project_platforms = ['All']
