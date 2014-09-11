#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# releasechk - a checker used for validation before releases
# Copyright (C) 2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Release checker to verify that everything is ready for release"""

import logging
import os
import sys
import subprocess
import urllib2
import zipfile

# prefer local cphct for package and version info

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path = [base_dir] + sys.path
from cphct import version_string 


fan_dir = 'fanbeam'
fan_base = os.path.join(base_dir, fan_dir)
cone_dir = 'conebeam'
cone_base = os.path.join(base_dir, cone_dir)
doc_dir = 'doc'
doc_base = os.path.join(base_dir, doc_dir)


def download_file(url, path):
    '''Downloads a file from url and saves it in path'''
    remote_fd = urllib2.urlopen(url)
    path_fd = open(path, 'wb')
    size = int(remote_fd.info()['content-length'])
    remaining_size = size
    block_size = 64*1024
    logging.info("downloading from %s" % url)
    while remaining_size:
        cur_size = min(block_size, remaining_size)
        data = remote_fd.read(cur_size)
        remaining_size -= cur_size
        path_fd.write(data)
        logging.debug("downloaded %d of %s bytes" % (size - remaining_size,
                                                     size))
    path_fd.close()
    remote_fd.close()

def unpack_file(zip_path, unpack_dst):
    '''Unpack a zip file with zip_path into unpack_dst'''
    zip_fd = zipfile.ZipFile(zip_path)
    for dst in zip_fd.namelist():
        if os.path.isabs(dst):
            # Avoid directory traversal issues/attacks
            logging.warning('skipping absolute path %s in zip!' % dst)
            return False
    zip_fd.extractall(unpack_dst)
    zip_fd.close()

def main(args):
    '''Run release validation checks'''
    log_level = logging.INFO
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    release_version = version_string
    working_dir = os.getcwd()
    download_map = {
        'circular/shepp-logan-1x32':
        'http://cphcttoolbox.googlecode.com/files/circular-shepp-logan-1x32.zip',
        'circular/shepp-logan-8x32':
        'http://cphcttoolbox.googlecode.com/files/circular-shepp-logan-8x32.zip',
        'spiral/shepp-logan-8x32':
        'http://cphcttoolbox.googlecode.com/files/spiral-shepp-logan-8x32.zip',
        }
    # Only check values are as expected without saving results to file
    check_map = {
        'npycenterfdk': {
            'app': os.path.join('fanbeam', 'centerfdk', 'npycenterfdk.py'),
            'cfg': os.path.join('circular', 'shepp-logan-1x32',
                                'shepp-logan-1x32-uint16-32x32x1-auto.cfg'),
            'opts': '--npy-save-output= --npy-postprocess-output=verify#%s' % \
            os.path.join('expected', 'npycenterfdk', 'circular', 'shepp-logan-1x32',
                         'shepp-logan-1x32-uint16-32x32x1-auto.res')
            },
        'npyfdk': {
            'app': os.path.join('conebeam', 'fdk', 'npyfdk.py'),
            'cfg': os.path.join('circular', 'shepp-logan-8x32',
                                'shepp-logan-8x32-uint16-32x32x32-auto.cfg'),
            'opts': '--npy-save-output= --npy-postprocess-output=verify#%s' % \
            os.path.join('expected', 'npyfdk', 'circular', 'shepp-logan-8x32',
                         'shepp-logan-8x32-uint16-32x32x32-auto.res')
            },
        'npykatsevich': {
            'app': os.path.join('conebeam', 'katsevich', 'npykatsevich.py'),
            'cfg': os.path.join('spiral', 'shepp-logan-8x32',
                                'shepp-logan-8x32-uint16-32x32x32-auto-0.5-rebin-16.cfg'),
            'opts': '--npy-save-output= --npy-postprocess-output=verify#%s' % \
            os.path.join('expected', 'npykatsevich', 'spiral', 'shepp-logan-8x32',
                         'shepp-logan-8x32-uint16-32x32x32-auto-0.5-rebin-16.res')
            }
        }
    check_map['cucenterfdk'] = check_map['npycenterfdk'].copy()
    check_map['cucenterfdk']['app'] = check_map['cucenterfdk']['app'].replace('npy', 'cu')
    check_map['cucenterfdk']['opts'] = check_map['cucenterfdk']['opts'].replace('npycenterfdk', 'cucenterfdk')
    check_map['clcenterfdk'] = check_map['npycenterfdk'].copy()
    check_map['clcenterfdk']['app'] = check_map['clcenterfdk']['app'].replace('npy', 'cl')
    check_map['clcenterfdk']['opts'] = check_map['clcenterfdk']['opts'].replace('npycenterfdk', 'clcenterfdk')
    check_map['cufdk'] = check_map['npyfdk'].copy()
    check_map['cufdk']['app'] = check_map['cufdk']['app'].replace('npy', 'cu')
    check_map['cufdk']['opts'] = check_map['cufdk']['opts'].replace('npyfdk', 'cufdk')
    check_map['clfdk'] = check_map['npyfdk'].copy()
    check_map['clfdk']['app'] = check_map['clfdk']['app'].replace('npy', 'cl')
    check_map['clfdk']['opts'] = check_map['clfdk']['opts'].replace('npyfdk', 'clfdk')
    check_map['cukatsevich'] = check_map['npykatsevich'].copy()
    check_map['cukatsevich']['app'] = check_map['cukatsevich']['app'].replace('npy', 'cu')
    check_map['cukatsevich']['opts'] = check_map['cukatsevich']['opts'].replace('npykatsevich', 'cukatsevich')
    # TODO: add once clkatsevich port is done
    #check_map['clkatsevich'] = check_map['npykatsevich'].copy()
    #check_map['clkatsevich']['app'] = check_map['clkatsevich']['app'].replace('npy', 'cl')
    #check_map['clkatsevich']['opts'] = check_map['clkatsevich']['opts'].replace('npykatsevich', 'clkatsevich')

    check_contents = ''
    for (app, app_conf) in check_map.items():
        check_contents += 'python %(app)s %(opts)s %(cfg)s\n' % app_conf

    print check_contents
        
    if args[1:]:
        release_version = args[1]

    if args[2:]:
        commands_path = args[2]
        try:
            check_fd = open(commands_path)
            check_commands = check_fd.read()
            check_fd.close()
        except IOError, ioe:
            logging.error("failed to load commands file %s: %s" % \
                          (commands_path, ioe))

    check_commands = check_contents.split('\n')

    logging.info("Running %s release checks" % release_version)
    if not release_version == version_string:
        logging.error("version mismatch: %s vs %s" % (release_version,
                                                      version_string))
        sys.exit(1)

    os.environ["PYTHONPATH"] = os.path.dirname(args[0])
    logging.info("Running with PYTHONPATH %(PYTHONPATH)s" % os.environ)

    # Download missing data

    for (target, url) in download_map.items():
        target_path = os.path.join(working_dir, target)
        zip_path = os.path.join(working_dir, os.path.basename(url))
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
            if not os.path.isfile(zip_path):
                download_file(url, zip_path)
            unpack_file(zip_path, working_dir)

    # Run checks
    for cmd in check_commands:
        check_proc = subprocess.Popen(cmd, shell=True)
        check_proc.wait()
        exit_code = check_proc.returncode
        if exit_code != 0:
            logging.error("check failed with exitcode %d" % exit_code)
            sys.exit(exit_code)

    logging.info("Validation for %s succeeded" % release_version)
    sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)
