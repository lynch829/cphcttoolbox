#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# setup - setuptools install helper
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

"""Setuptools install helper"""

import os
import sys
import subprocess
from setuptools import setup, find_packages
from distutils.command.sdist import sdist as DistutilsSdist


# Build docs first if called with sdist command
# http://stackoverflow.com/questions/1754966/how-can-i-run-a-makefile-in-setup-py

class DocUpdatingSdist(DistutilsSdist):

    """SDist handler that builds docs first"""

    def run(self):
        """Override default sdist step to build docs first"""

        subprocess.check_call(['make', '-C', doc_src, 'all'])
        DistutilsSdist.run(self)


# prefer local cphct for package and version info

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path = [base_dir] + sys.path
from cphct import version_string, short_name, project_team, \
    project_email, short_desc, long_desc, project_url, download_url, \
    license_name, project_class, project_keywords, versioned_requires, \
    project_requires, project_extras, project_platforms, maintainer_team, \
    maintainer_email

# IMPORTANT: expected to installed from release made with ./setup.py sdist
# thus no filtering of enabled algos or engines as MANIFEST should already
# include/exclude the relevant ones from source dist.

fan_dir = 'fanbeam'
fan_base = os.path.join(base_dir, fan_dir)
cone_dir = 'conebeam'
cone_base = os.path.join(base_dir, cone_dir)
doc_dir = 'doc'
doc_base = os.path.join(base_dir, doc_dir)
doc_src = 'doc-src'

exec_modules = []
for base in [fan_base, cone_base]:
    for (root, _, files) in os.walk(base):
        rel_dir = root.replace(base_dir + os.sep, '')
        for name in [i for i in files if i.endswith('.py')]:
            exec_modules.append(os.path.join(rel_dir, name))

core_scripts = ['%s' % i for i in exec_modules]
example_scripts = []
example_scripts_paths = [os.path.join('examples', '%s.py' % i) for i in
                         example_scripts]
extra_docs = os.listdir(doc_base)
extra_docs_paths = [os.path.join(doc_dir, name) for name in extra_docs]

# Distro packaging name

install_name = 'python-%s' % short_name

if __name__ == '__main__':

    setup(
        name=short_name,
        version=version_string,
        description=short_desc,
        long_description=long_desc,
        author=project_team,
        author_email=project_email,
        maintainer=maintainer_team,
        maintainer_email=maintainer_email,
        url=project_url,
        download_url=download_url,
        license=license_name,
        classifiers=project_class,
        keywords=project_keywords,
        platforms=project_platforms,
        install_requires=versioned_requires,
        requires=project_requires,
        extras_require=project_extras,
        packages=find_packages(),
        # Include any *.cu or *.cl kernel files found inside all packages
        package_data={'': ['*.cu', '*.cl']},
        include_package_data=True,
        scripts=core_scripts,
        data_files=[(os.path.join('share', 'doc', install_name),
                    extra_docs_paths), (os.path.join('share', 'doc',
                    install_name, 'examples'), example_scripts_paths),
                    ],
        cmdclass={'sdist': DocUpdatingSdist},
        )
