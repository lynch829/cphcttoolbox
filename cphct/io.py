#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - input/ouput helpers
# Copyright (C) 2011-2012  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Input/output helper functions"""

import gc
import os
import tempfile

(PATH, ANGLE, FILTERED) = ('path', 'angle', 'filtered')

(XMIN, YMIN, ZMIN) = ('xmin', 'ymin', 'zmin')
(XMAX, YMAX, ZMAX) = ('xmax', 'ymax', 'zmax')
UNSET = '__UNSET__'

DEFAULT_READ_FIELDS = [PATH, ANGLE]
DEFAULT_WRITE_FIELDS = [
    PATH,
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    ZMIN,
    ZMAX,
    ]


def create_path_dir(path):
    """Create the directory part of *path* if it doesn't already exist

    Parameters
    ----------
    path : str
       Complete file or directory path

    Returns : str
       The created path
    """

    dir_path = os.path.split(path)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def expand_path(working_directory, path):
    """Expand relative paths to absolute paths.
    If *path* is not absolute we prefix with *working_directory*.

    Parameters
    ----------
    working_directory : str
        Working directory path to prefix relative paths with.
    path : str
        relative or absolute path.

    Returns
    -------
    output : str
        The path expanded with working_directory prefix if relative
        and just the original path otherwise.
    """

    # join automatically ignores leading args if last path is absolute

    return os.path.join(working_directory, path)


def path_expander(conf, key):
    """Inline configuration path expander to prefix certain paths with the
    working_directory value there.

    Parameters
    ----------
    conf : dict
        Configuration dictionary to be modified.
    key : str
        Name of path variable in configuration to expand.

    Returns
    -------
    output : dict
        Returns the same configuration dictionary with the value for path key
        expanded with working_directory.
    """

    if conf.get(key, None):
        conf[key] = expand_path(conf['working_directory'], conf[key])
    return conf


def collapse_path(working_directory, path):
    """Collapse absolute paths to relative paths.
    If *path* is absolute we strip *working_directory* and otherwise we leave
    it alone.
    
    ----------
    working_directory : str
        Working directory path to strip from absolute paths.
    path : str
        relative or absolute path.

    Returns
    -------
    output : str
        The path collapsed by working_directory prefix if absolute and just
        the original path otherwise.
    """

    return os.path.relpath(path, working_directory)


def extract_path(path_or_fd):
    """Extract path from open file object or path string.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or path to file with binary projections.

    Returns
    -------
    output : str
        Returns the path string for *path_or_fd*.
    """

    if isinstance(path_or_fd, basestring):
        return path_or_fd
    else:
        return path_or_fd.name


def create_scene_entry(
    path=UNSET,
    xmin=UNSET,
    xmax=UNSET,
    ymin=UNSET,
    ymax=UNSET,
    zmin=UNSET,
    zmax=UNSET,
    ):
    """
    Create scene dictionary entry from args that are not UNSET

    Parameters
    ----------
    path : str
        Projection path.
    xmin : float
        lowest X coordinate
    xmax : float
        highest X coordinate
    ymin : float
        lowest Y coordinate
    ymax : float
        highest Y coordinate
    zmin : float
        lowest Z coordinate
    zmax : float
        highest Z coordinate

    Returns
    -------
    output : dict
        Dictionary for a single scene entry containing the provided arguments
        and not the ones left unset. Entries are ready for use in scene
        writing.
    """

    raw_entry = {
        PATH: path,
        XMIN: xmin,
        XMAX: xmax,
        YMIN: ymin,
        YMAX: ymax,
        ZMIN: zmin,
        ZMAX: zmax,
        }
    entry = {}
    for (key, val) in raw_entry.items():
        if val != UNSET:
            entry[key] = val
    return entry


def read_scene(path, working_directory, fields=DEFAULT_READ_FIELDS):
    """Read and parse scene description from simple format CSV file in path.
    Each line in CSV file to be parsed should be on the format specified by
    the fields. For the case of fields=[PATH, ANGLE] that means:
    path, angle
    e.g.
    proj000.jpg, 0.1221
    ...
    proj359.jpg, 359.8937

    Comments initialised with a '#' are ignored and so are lines that do not
    fit the fields specification.

    The parsed result is a sorted list of dictionaries with one entry for each
    line, where each of the fields from available_fields are set to their raw
    string values.
    Thus for the example above it would be something like:
    [{PATH: 'proj000.jpg', ANGLE: '0.1221'},
    ... ,
    {PATH: 'proj359.jpg', ANGLE: '359.8937'}]

    Parameters
    ----------
    path : str
        Scene path.
    working_directory : str
        Working directory path to prefix relative scene paths with.
    fields : list of str
        Ordered list of field names to read

    Returns
    -------
    output : list of dict
        List of scene entry dictionaries. The list will contain the entries
        in the order they were read from the scene file and entries contain a
        mapping from the field names to the parsed values.
    """

    scene_fd = open(path, 'r')
    parsed = []
    for line in scene_fd:

        # ignore comments

        line = line.split('#', 1)[0]
        if not line:
            continue
        parts = line.split(',')

        # ignore broken lines

        if len(parts) < len(fields):
            continue

        # ignore extra fields

        parts = parts[:len(fields)]
        entry = {}
        for (name, val) in zip(fields, parts):
            entry[name] = val.strip()

        # Expand relative projection paths

        if PATH in entry:
            entry[PATH] = expand_path(working_directory, entry[PATH])

        parsed.append(entry)

    scene_fd.close()
    return parsed


def write_scene(
    path,
    scene,
    fields=DEFAULT_WRITE_FIELDS,
    header_lines=[],
    ):
    """Write scene description from sorted list of dictionares in scene to
    simple format CSV file in given path.
    Each dictionary in scene list should contain at least the names in fields
    and the function will write a line of comma separated field values for
    each entry.
    for a scene list like
    [{PATH: 'zslice000.jpg', XMIN: 0, XMAX: 512, YMIN: 0, YMAX: 512, ZMIN: 0,
      ZMAX: 1},
    ... ,
    {PATH: 'zslice511.jpg', XMIN: 0, XMAX: 512, YMIN: 0, YMAX: 512, ZMIN: 511,
      ZMAX: 512}]
    with the default fields this means lines like:
    zslice000.jpg, 0, 512, 0, 512, 0, 1
    ...
    zslice511.jpg, 0, 512, 0, 512, 511, 512

    The optional header_lines argument is a list of strings to insert as
    comments at the top of the CSV file.

    Parameters
    ----------
    path : str
        Scene path.
    scene : list of dict
        Working directory path to prefix relative paths with.
    fields : list of str
        Ordered list of field names to read
    header_lines : list of str
        Ordered list of header lines to write at the top of the scene file
    """

    scene_fd = open(path, 'w')
    lines = ['# %s\n' % line for line in header_lines]
    for entry in scene:
        parts = []

        # Add string values

        for name in fields:
            if name in entry:
                parts.append(str(entry[name]))

        # skip broken entries

        if len(parts) != len(fields):
            continue
        lines.append('%s\n' % ', '.join(parts))
    scene_fd.writelines(lines)
    scene_fd.close()


def temporary_file(conf, mode='w+b', named=False):
    """Return a tempfile.TemporaryFile(*mode*, dir=tmp_dir) file object where
    tmp_dir is temporary_directory from *conf* dictionary.
    If named is set the result is the corresponding
    tempfile.NamedTemporaryFile(*mode*, dir=tmp_dir) instead.

    Parameters
    ----------
    conf : dict
        Configuration dictionary to be filled.
    mode : str
        Access mode string as known from open().
    name : bool
        If set the returned file object is guaranteed to have a visible file
        name.

    Returns
    -------
    output : obj
        A (Named)TemporaryFile object using provided mode and temporary
        directory from conf.
    """

    if named:
        tmp_func = tempfile.NamedTemporaryFile
    else:
        tmp_func = tempfile.TemporaryFile
    return tmp_func(mode=mode, dir=conf['temporary_directory'])


def fill_base_conf(conf):
    """Remaining configuration after handling command line options.

    Parameters
    ----------
    conf : dict
        Configuration dictionary to be filled.

    Returns
    -------
    output : dict
        Filled configuration dictionay where all the base configuration
        options are set.
        Adds cphcttoolbox_base entry pointing to actual cphct install path
        for use in any additional location specific initialization.
    """

    conf['app_state'] = {}
    conf['app_state']['projs'] = {}
    conf['app_state']['backproject'] = {}

    conf['cphcttoolbox_base'] = \
        os.path.dirname(os.path.dirname(__file__))

    path_expander(conf, 'temporary_directory')

    return conf


def engine_alloc(
    conf,
    alloc_name,
    key,
    data,
    size,
    ):
    """Stores a {'data': data, 'size': size} entry with given key in
    conf[alloc_name] .
    Used to keep track of allocated engine-specific data and provide shared
    access.

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    alloc_name : str
        Variable name for engine-specific alloc data
    key : str
        Variable name for *data*
    data : object
        Allocated data array
    size : int
        Size of allocated data array

    Returns
    -------
    output : dict
        Same configuration dictionary where the conf[alloc_name] dictionary is
        extended to map *key* to a new dictionary with *data* and *size*.

    Raises
    ------
    ValueError
        If *key* is *None* or
        if *key* is already in allocated set
    """

    if key is None:
        raise ValueError('Provided key is \'None\'')
    elif key in conf[alloc_name]:
        msg = 'Engine data element with key: %s already allocated' % key
        raise ValueError(msg)

    conf[alloc_name][key] = {'data': data, 'size': size}
    return conf


def get_engine_data(conf, alloc_name, key):
    """Extracts engine-specific data for variable *key*

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    alloc_name : str
        Variable name for engine-specific alloc data
    key : str
        Data entry dict key

    Returns
    -------
    output : object
        Data entry corresponding to *key* in conf[alloc_name]
        *None* if *key* is not in conf[alloc_name]
    """

    out = None
    if key in conf[alloc_name]:
        out = conf[alloc_name][key]['data']
    return out


def get_engine_size(conf, alloc_name, key):
    """Extracts engine-specific data size for variable *key*

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    alloc_name : str
        Variable name for engine-specific alloc data
    key : str
        Data entry dict key

    Returns
    -------
    output : int
        Size entry corresponding to *key* in conf[alloc_name]
        Zero if *key* is not in conf[alloc_name]
    """

    out = 0
    if key in conf[alloc_name]:
        out = conf[alloc_name][key]['size']
    return out


def get_engine_total_size(conf, alloc_name):
    """Extracts the total size of allocated engine-specific data

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    alloc_name : str
        Variable name for engine-specific alloc data

    Returns
    -------
    output : int
        Sum of all size entries in conf[alloc_name]
    """

    return sum([entry['size'] for entry in conf[alloc_name].values()])


def engine_free(
    conf,
    alloc_name,
    key,
    ignore_missing,
    ):
    """Free engine-specific data entry *key* from
    conf[alloc_name]

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    alloc_name : str
        Variable name for engine-specific alloc data
    key : str
        Variable to be freed
    ignore_missing : bool
        If *True* the engine-specific data for variable *key* is freed if
        present.
        If *False* exceptions may be raised based on the value of *key*
        (see Raises),

    Returns
    -------
    output : dict
        Configuration dictionary

    Raises
    ------
    ValueError
        If *key* is None or
        if *key* isn not in allocated set
    """

    if not ignore_missing:
        if key is None:
            raise ValueError('Provided key is \'None\'')
        elif not key in conf[alloc_name]:
            msg = 'Provided variable \'%s\'' % key
            msg = '%s is not in %s data' % (msg, alloc_name)
            raise ValueError(msg)

    if key is not None and key in conf[alloc_name]:
        del conf[alloc_name][key]
    return conf


def engine_free_all(conf, alloc_name, garbage_collect):
    """Free all engine-specific data entries from conf[alloc_name].

    Optionally explicitly trigger garbage collection to free any related
    memory allocations.

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    alloc_name : str
        Variable name for engine-specific alloc data

    Returns
    -------
    output : dict
        Configuration dictionary
    """

    for key in conf[alloc_name].keys():
        engine_free(conf, alloc_name, key, False)
    if garbage_collect:
        gc.collect()
    return conf


