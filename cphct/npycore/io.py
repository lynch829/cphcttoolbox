#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# io - numpy core specific input/ouput helpers
# Copyright (C) 2011-2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Numpy core specific input/output helper functions"""

import os
from cphct.io import fill_base_conf, expand_path, read_scene, PATH, \
    ANGLE, FILTERED, DEFAULT_READ_FIELDS
from cphct.io import engine_alloc, get_engine_data, get_engine_size, \
    get_engine_total_size, engine_free, engine_free_all, extract_path
from cphct.npycore import allowed_data_types, pi, zeros, array, \
    ndarray, uint8, uint16, float32, fromfile, loadtxt, load, savetxt, \
    save, savez, cast, arcsin
from cphct.npycore.misc import size_from_shape

auto_extensions = {
    'image': ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.ppm', '.pgm',
              '.pbm', '.gif'],
    'text': ['.txt', '.gz'],
    'native': ['.npy', '.npz'],
    'data': ['.bin', '.raw']
    }


def load_scene_chunk(
    first,
    last,
    working_directory,
    path,
    detector_rows,
    detector_columns,
    in_type=uint16,
    out_type=float32,
    skip_projs=True,
    scene_fields=DEFAULT_READ_FIELDS,
    ):
    """Load scene description from simple format CSV file in path, using the
    parser from the general io module.
    If skip_projs is set the scene images will be ignored and None
    returned instead of the projections.
    If the scene contains proj files of unsigned int type the values will be
    automatically transformed from intensity values to air normalized
    attenuation.
    Please note that the intensity projections may equal the attenuation
    projections if the loaded projections are already converted to the
    attenuation form.

    Parameters
    ----------
    first : int
        Index of first projection to load.
    last : int
        Index of last projection to load.
    working_directory : str
        Working directory path to prefix relative scene paths with.
    path : str
        Scene path.
    detector_rows : int
        Number of rows in each projection.
    detector_columns : int
        Number of columns in each projection.
    in_type : dtype
        Raw input i.e. projection data type.
    out_type : dtype
        Output i.e. wanted projection data type.
    skip_projs : bool
        Flag to specify that scene projection files should be ignored to use a
        single binary projection stack instead.
    scene_fields : list of str
        Ordered list of field names to read from scene file.

    Returns
    -------
    output : (dict, dict, dict)
        Returns a 3-tuple with the requested chunk read from a scene.
        It contains the list of parsed entries, a detector_rows X
        detector_columns matrix of intensity projections and a detector_rows X
        detector_columns matrix of attenuation projections. All three will
        contain the scene entries in the order they were read from the scene
        file and entries contain a mapping from the field names to the parsed
        values.
    """

    parsed = read_scene(path, working_directory, fields=scene_fields)
    if last >= 0:
        parsed = parsed[:last + 1]
    if first >= 0:
        parsed = parsed[first:]
    (rows, cols) = (detector_rows, detector_columns)
    projections = zeros((len(parsed), rows, cols), dtype=out_type)
    if not skip_projs:
        i = 0
        for entry in parsed:

            # projection files are in_type image or raw files

            projections[i] = load_auto(entry[PATH], (rows, cols),
                    in_type)
            i += 1
    return (parsed, projections)


def load_scene(
    working_directory,
    path,
    detector_rows,
    detector_columns,
    in_type=uint16,
    out_type=float32,
    skip_projs=True,
    scene_fields=DEFAULT_READ_FIELDS,
    ):
    """Load scene description from simple format CSV file in path, using the
    parser from the general io module.
    If skip_projs is set the scene images will be ignored and None
    returned instead of the projections.
    This is a simple wrapper for load_scene_chunk using the entire set of
    projections.

    Parameters
    ----------
    working_directory : str
        Working directory path to prefix relative scene paths with.
    path : str
        Scene path.
    detector_rows : int
        Number of rows in each projection.
    detector_columns : int
        Number of columns in each projection.
    in_type : dtype
        Raw input i.e. projection data type.
    out_type : dtype
        Output i.e. wanted projection data type.
    skip_projs : bool
        Flag to specify that scene projection files should be ignored to use a
        single binary projection stack instead.
    scene_fields : list of str
        Ordered list of field names to read from scene file.

    Returns
    -------
    output : (dict, dict, dict)
        Returns a 3-tuple with all the projections read from a scene.
        It contains the list of parsed entries, a detector_rows X
        detector_columns matrix of intensity projections and a detector_rows X
        detector_columns matrix of attenuation projections. All three will
        contain the scene entries in the order they were read from the scene
        file and entries contain a mapping from the field names to the parsed
        values.
    """

    return load_scene_chunk(
        -1,
        -1,
        working_directory,
        path,
        detector_rows,
        detector_columns,
        in_type,
        out_type,
        skip_projs,
        scene_fields,
        )


def load_projs(
    path_or_fd,
    shape,
    in_type,
    out_type,
    items=-1,
    ):
    """Load binary projections dumped as dtype in file

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or path to file with binary projections.
    shape : tuple of int
        Shape of projections matrix.
    in_type : dtype
        Raw input i.e. projection data type.
    out_type : dtype
        Output i.e. wanted projection data type.
    items : intbool
        Number of items to read for partial reading of projection files.

    Returns
    -------
    output : ndarray
        Returns an array of the requested shape with the loaded projections.

    Raises
    -------
    ValueError
        If the loaded projections do not fit the requested shape.
    """

    projections = fromfile(path_or_fd, dtype=in_type, count=items)
    try:
        projections.shape = shape
    except ValueError, vae:
        raise ValueError('Failed to load from %s - shape mismatch %s (%s)'
                          % (path_or_fd, projections.shape, shape))
    if in_type != out_type:
        projections = projections.astype(out_type)
    return projections


def load_projs_chunk(
    first,
    last,
    working_directory,
    scene_path,
    projs_path,
    detector_rows,
    detector_columns,
    in_type,
    out_type,
    prefiltered_projs=False,
    scene_fields=DEFAULT_READ_FIELDS,
    ):
    """Load projections in range first to last inclusive

    Parameters
    ----------
    first : int
        Index of first projection to load.
    last : int
        Index of last projection to load.
    working_directory : str
        Working directory path to prefix relative scene paths with.
    scene_path : str
        Scene path.
    projs_path : str
        Separate projections path that is used to override scene projs if set.
    detector_rows : int
        Number of rows in each projection.
    detector_columns : int
        Number of columns in each projection.
    in_type : dtype
        Raw input i.e. projection data type.
    out_type : dtype
        Output i.e. wanted projection data type.
    prefiltered_projs : bool
        Optional marker to point out that loaded projections are already on
        prefiltered form. The projection meta data will include this
        information so that the prefiltering step is skipped later.
    scene_fields : list of str
        Ordered list of field names to read from scene file.

    Returns
    -------
    output : (dict, dict, dict)
        Returns a 3-tuple with the requested chunk read from a scene.
        It contains the list of parsed entries, a detector_rows X
        detector_columns matrix of intensity projections and a detector_rows X
        detector_columns matrix of attenuation projections. All three will
        contain the scene entries in the order they were read from the scene
        file and entries contain a mapping from the field names to the parsed
        values. An extra scene field to mark prefiltering status is added to
        the parsed entries.

    Raises
    -------
    ValueError
        If called without scene_path.
    """

    # Last index is inclusive

    if scene_path:

        # Load possibly non-uniform offsets here along with angles

        (projs_list, projs) = load_scene_chunk(
            first,
            last,
            working_directory,
            scene_path,
            detector_rows,
            detector_columns,
            in_type,
            out_type,
            skip_projs=projs_path,
            scene_fields=scene_fields,
            )

        # Extend projs_list with prefiltering information

        for entry in projs_list:
            entry[FILTERED] = prefiltered_projs

        source_angles = [entry[ANGLE] for entry in projs_list]
        if projs_path:

            # Load only the part requested

            projs_fd = open(projs_path, 'r')
            projs_shape = (len(source_angles), detector_rows,
                           detector_columns)

            # include last

            chunk_shape = (1 + last - first, projs_shape[1],
                           projs_shape[2])
            first_offset = first * projs_shape[1] * projs_shape[2]
            chunk_items = chunk_shape[0] * chunk_shape[1] \
                * chunk_shape[2]
            projs_fd.seek(in_type(0).nbytes * first_offset)

            projs = load_projs(projs_fd, chunk_shape, in_type,
                               out_type, chunk_items)

            projs_fd.close()
    else:
        raise ValueError('load projs chunk called without scene path')
    return (projs_list, projs)


def load_data(path_or_fd, shape, data_type):
    """Load next ELEMS *data_type* values from *path_or_fd* file object or path
    and return the values as a matrix with dimensions specified in *shape*:
    *path_or_fd* must contain binary data and the bytes are loaded into a numpy
    array of provided *data_type* with provided dimensions.
    Only ELEMS values will be read so that the function can be called
    repeatedly to load stacked data.
    ELEMS is calculated as the product of the *shape* tuple entries, i.e. the
    number of elements in such a matrix.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    shape : tuple of int
        Shape of returned matrix.
    data_type : dtype
        Output matrix data type.

    Returns
    -------
    output : ndarray
        Returns an array loaded from *path_or_fd* and with requested shape and
        data type. If *path_or_fd* is a file object the cursor is moved ELEMS
        times size of *data_type* bytes forward, with ELEMS is defined as
        described above.

    Raises
    -------
    ValueError
        If called with a shape that doesn't fit the loaded data.
    """

    elems = size_from_shape(shape)
    values = fromfile(path_or_fd, count=elems, dtype=data_type)
    try:
        values.shape = shape
    except ValueError, vae:
        raise ValueError('Failed to load data from %s - shape mismatch %s (%s)'
                          % (path_or_fd, values.shape, shape))
    return values


def load_text(path_or_fd, shape, data_type):
    """Load saved (optionally compressed) text data matrix from *path_or_fd*:
    file must contain values saved in numpy.savetxt format and they are
    loaded into a numpy array with provided shape and dtype.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    shape : tuple of int
        Shape of returned matrix.
    data_type : dtype
        Output matrix data type.

    Returns
    -------
    output : ndarray
        Returns an array loaded from *path_or_fd* and with requested shape and
        *data_type*. If *path_or_fd* is a file object the cursor is moved ELEMS
        times size of *data_type* bytes forward. Where ELEMS is the number of
        elements in a matrix with given shape.

    Raises
    -------
    ValueError
        If called with a shape that doesn't fit the loaded data.
    """

    values = loadtxt(path_or_fd, dtype=data_type)
    try:
        values.shape = shape
    except ValueError, vae:
        raise ValueError('Failed to load from %s - shape mismatch %s (%s)'
                          % (path_or_fd, values.shape, shape))
    return values


def load_native(path_or_fd, shape, data_type):
    """Load saved (optionally compressed) native data matrix from *path_or_fd*:
    file must contain values saved in numpy.save(z) format and they are
    loaded into a numpy array with provided shape and dtype.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    shape : tuple of int
        Shape of returned matrix.
    data_type : dtype
        Output matrix data type.

    Returns
    -------
    output : ndarray
        Returns an array loaded from *path_or_fd* and with requested shape and
        *data_type*. If *path_or_fd* is a file object the cursor is moved ELEMS
        times size of *data_type* bytes forward. Where ELEMS is the number of
        elements in a matrix with given shape.

    Raises
    -------
    ValueError
        If called with a shape that doesn't fit the loaded data.
    """

    values = load(path_or_fd)

    # load() on a savez file returns a dict-like object of names and arrays

    try:
        values = values[values.keys()[0]]
    except:
        pass
    values = data_type(values)
    try:
        values.shape = shape
    except ValueError, vae:
        raise ValueError('Failed to load from %s - shape mismatch %s (%s)'
                          % (path_or_fd, values.shape, shape))
    return values


def load_image(path_or_fd, shape, data_type, dynamic_range=None):
    """Try hard to load saved image from *path_or_fd* using any available
    helper tool. The values are loaded into a numpy array of the requested
    shape and *data_type*.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    shape : tuple of int
        Shape of returned matrix.
    data_type : dtype
        Output matrix data type.
    dynamic_range : 2-tuple of ints, optional
        Dynamic range of loaded image (min, max)

    Returns
    -------
    output : ndarray
        Returns an array loaded from image in *path_or_fd* and with requested
        shape and *data_type*. If *path_or_fd* is a file object the cursor is
        moved ELEMS times size of *data_type* bytes forward. Where ELEMS is
        the number of elements in a matrix with given shape.

    Raises
    -------
    ValueError
        If called with an image path that can't be loaded with any of the
        helper tools or if called with a shape that doesn't fit the loaded
        data.
    """

    got_values = False
    path = extract_path(path_or_fd)

    # We manually convert 16-bit tif data later

    manual_convert = False
    convert_max = 255
    if path.endswith('tif') or path.endswith('tiff'):
        manual_convert = True
    try:
        if manual_convert:
            raise ImportError('Do not use scipy for tif files!')
        import scipy.misc
        values = scipy.misc.imread(path_or_fd)
        got_values = True
    except ImportError:
        pass
    try:

        # PIL wrapper borrowed from scipy implementation

        from PIL import Image
        img = Image.open(path_or_fd)
        if not Image.isImageType(img):
            raise TypeError('Input is not a PIL image.')
        if manual_convert:

            # Convert 16-bit tif to integers to avoid loosing data

            img = img.convert('I')
        else:

            # Convert RGB color lists to a single grey scale value

            img = img.convert('F')
        values = array(img)
        got_values = True
    except ImportError:
        pass
    if not got_values:
        raise ValueError('no suitable image loading tools available!')
    if dynamic_range:
        (cmin, cmax) = (dynamic_range[0], dynamic_range[1])
        values *= (1.0 * (cmax - cmin)) / convert_max
        values += cmin
    values = data_type(values)
    try:
        values.shape = shape
    except ValueError, vae:
        raise ValueError('Failed to load from %s - shape mismatch %s (%s)'
                          % (path_or_fd, values.shape, shape))
    return values


def load_auto(path_or_fd, shape, data_type, dynamic_range=None):
    """Load saved data matrix or image from *path_or_fd*:
    Detect type and call appropriate load back end.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    shape : tuple of int
        Shape of returned matrix.
    data_type : dtype
        Output matrix data type.
    dynamic_range : 2-tuple of ints, optional
        Dynamic range of loaded image (min, max)
        used when loading images like '.png', '.jpg', '.bmp', etc.

    Returns
    -------
    output : ndarray
        Returns an array loaded from *path_or_fd* and with requested shape.

    Raises
    -------
    ValueError
        If called with a shape that doesn't fit the loaded data.
    """

    path = extract_path(path_or_fd)
    file_ext = os.path.splitext(path)[1].lower()
    if file_ext in auto_extensions['image']:
        return load_image(path_or_fd, shape, data_type, dynamic_range)
    elif file_ext in auto_extensions['text']:
        return load_text(path_or_fd, shape, data_type)
    elif file_ext in auto_extensions['native']:
        return load_native(path_or_fd, shape, data_type)
    elif file_ext in auto_extensions['data']:
        return load_data(path_or_fd, shape, data_type)
    else:
        print "WARNING: Unexpected file extension, %s: load as raw data" % \
              file_ext
        return load_data(path_or_fd, shape, data_type)


def save_data(path_or_fd, values):
    """Save raw matrix values to *path_or_fd*:
    takes a numpy array as values and dumps a single long string of bytes.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    values : ndarray
        Data array of any shape to be saved.
    """

    values.tofile(path_or_fd)


def save_text(path_or_fd, values):
    """Save matrix values as (optionally compressed) text to *path_or_fd*:
    takes a numpy array as values and saves in text format suitable for
    the load_text function to use.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    values : ndarray
        Data array of any shape to be saved.
    """

    savetxt(path_or_fd, values)


def save_native(path_or_fd, values):
    """Save matrix values as (optionally compressed) native data to
    *path_or_fd*:
    takes a numpy array as values and saves in native format suitable for
    the load_native function to use.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    values : ndarray
        Data array of any shape to be saved.
    """

    path = extract_path(path_or_fd)
    if path.endswith('.npz'):
        savez(path_or_fd, values)
    else:
        save(path_or_fd, values)


def save_image(path_or_fd, values, dynamic_range=None):
    """Try hard to save values as an image in *path_or_fd* using any available
    helper tool.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    values : ndarray
        Image array any 2-D shape to be saved.
    dynamic_range : 2-tuple of ints, optional
        Dynamic range of saved image (min, max)
    Raises
    -------
    ValueError
        If called with an image path (type) that can't be saved with any of the
        helper tools.
    """

    path = extract_path(path_or_fd)

    if dynamic_range:
        (cmin, cmax) = (dynamic_range[0], dynamic_range[1])
    else:
        (cmin, cmax) = (values.min(), values.max())

    try:

        # PIL wrapper borrowed from scipy implementation

        from PIL import Image

        (low, high) = (0, 255)
        scale = high * 1. / (cmax - cmin or 1)
        bytedata = ((values * 1. - cmin) * scale + 0.4999).astype(uint8)
        bytedata += cast[uint8](low)

        # shape is opposite of wanted image layout

        shape = values.shape[::-1]
        img = Image.fromstring('L', shape, bytedata.tostring())
        img.save(path_or_fd)
        return
    except ImportError:
        pass
    try:

        # Plain scipy - please note that this import may leak GPU memory!!!

        import scipy.misc

        scipy.misc.toimage(values, cmin=cmin,
                           cmax=cmax).save(path_or_fd)
        return
    except ImportError:
        pass
    raise ValueError('gave up saving image to %s - no suitable tools'
                     % path)


def save_auto(path_or_fd, values, dynamic_range=None):
    """Save values matrix in format given by file extension in *path_or_fd*:
    Detect type and call appropriate save back end.

    Parameters
    ----------
    path_or_fd : file or str
        Open file object or file path.
    values : ndarray
        Data array of any shape to be saved.
    dynamic_range : 2-tuple of ints, optional
        Dynamic range of saved image (min, max),
        used when saving images like '.png', '.jpg', '.bmp', etc.
    """

    path = extract_path(path_or_fd)

    # Make dir if it doesn't exist already

    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass
    file_ext = os.path.splitext(path)[1].lower()
    if file_ext in auto_extensions['image']:
        return save_image(path_or_fd, values, dynamic_range)
    elif file_ext in auto_extensions['text']:
        return save_text(path_or_fd, values)
    elif file_ext in auto_extensions['native']:
        return save_native(path_or_fd, values)
    elif file_ext in auto_extensions['data']:
        return save_data(path_or_fd, values)
    else:
        print "WARNING: Unexpected file extension, %s: save as raw data" % \
              file_ext
        return save_data(path_or_fd, values)


def load_helper_proj(helper, conf, idt):
    """
    Gets numpy helper projection based on the user provided *helper* value. If
    it is a constant a projection of dimensions from conf detector_rows and
    detector_columns is created and if it is a filename the projection there
    is loaded.

    Parameters
    ----------
    helper : str
        Constant value or file path
    conf : dict
        A dictionary of configuration options.
    idt : dtype
        Input matrix data type.

    Returns
    -------
    output : ndarray
        Returns loaded or constant matrix of dtype *idt*.

    Raises
    ------
    ValueError
        If provided *helper* value is neither a suitable file nor a single
        value compatible with idt.
    """

    helper_path = expand_path(conf['working_directory'], helper)

    helper_shape = (conf['detector_rows'], conf['detector_columns'])

    if os.path.isfile(helper_path):
        try:
            helper_matrix = load_auto(helper_path, helper_shape, idt)
        except Exception, msg:
            raise ValueError('Loading helper proj file: %s' % msg)
    else:
        helper_val = 0
        if helper != '':
            try:
                helper_val = idt(helper)
            except Exception, msg:
                raise ValueError('Invalid helper proj value: %s' % msg)

        helper_matrix = zeros(helper_shape, idt) + helper_val

    return helper_matrix


def fill_base_npycore_conf(conf):
    """Remaining default configuration after handling command line options.
    Casts all floating point results using float data type from conf.
    This version is for the shared numpy core.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary filled with numpy core settings.
    """

    fill_base_conf(conf)

    # Translate precision to numpy data types once and for all

    adt = allowed_data_types
    idt = conf['input_data_type'] = adt[conf['input_precision']]
    fdt = conf['data_type'] = adt[conf['precision']]
    odt = conf['output_data_type'] = adt[conf['output_precision']]

    # Make sure all float values get the right precision before we continue

    for (key, val) in conf.items():
        if isinstance(val, float):
            conf[key] = fdt(val)

    # Initiate dict for numpy allocations

    conf['npy_data'] = {}

    # Kernel helpers

    conf['float_size'] = fdt(42.0).nbytes
    conf['pi'] = pi

    # Set up additional vars based on final conf

    conf['x_len'] = fdt(conf['x_max'] - conf['x_min'])
    conf['y_len'] = fdt(conf['y_max'] - conf['y_min'])

    # Important: by definition x, y voxels start and end with *center* at
    # *_min and *_max values.
    # I.e. the boundary voxels occuppy only *half* a voxel inside the range
    # [*_min, *_max] leaving delta_* slightly bigger than *_len/*_voxs .
    # (one might argue that *_min is really min plus delta_*/2, but anyway)

    conf['delta_x'] = fdt(conf['x_len'] / (conf['x_voxels'] - 1))
    conf['delta_y'] = fdt(conf['y_len'] / (conf['y_voxels'] - 1))

    conf['total_projs'] = int(conf['total_turns']
                              * conf['projs_per_turn'])

    # ## Additional shared settings for fan beam algos here

    conf['fov_diameter'] = fdt(max(conf['x_len'], conf['y_len']))
    conf['fov_radius'] = fdt(0.5 * conf['fov_diameter'])
    conf['scan_diameter'] = fdt(conf['source_distance']
                                + conf['detector_distance'])
    conf['scan_radius'] = fdt(0.5 * conf['scan_diameter'])
    conf['half_fan_angle'] = fdt(arcsin(conf['fov_radius']
                                 / conf['scan_radius']))

    # Calculate generic pixel size for case without auto sizing

    if conf['detector_pixel_width'] > 0.0:
        conf['detector_width'] = fdt(1. * conf['detector_pixel_width']
                * conf['detector_columns'])

        # Center of pixel in detector_half_width

        conf['detector_half_width'] = fdt(0.5 * (conf['detector_width']
                - conf['detector_pixel_width']))
    elif conf['detector_width'] > 0.0:

        conf['detector_pixel_width'] = fdt(1. * conf['detector_width']
                / conf['detector_columns'])

        # Center of pixel in detector_half_width

        conf['detector_half_width'] = fdt(0.5 * (conf['detector_width']
                - conf['detector_pixel_width']))

    return conf


def npy_alloc(
    conf,
    key,
    data,
    size=None,
    ):
    """Stores a {'data': data, 'size': size} entry with given key in
    conf['npy_data'] .
    Used to keep track of allocated numpy data and provide shared access.

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    key : str
        Variable name for *data*
    data : ndarray
        Allocated data array
    size : int
        Size of allocated data array (optional: defaults to data.nbytes)

    Returns
    -------
    output : dict
        Same configuration dictionary where the conf['npy_data'] dictionary is
        extended to map *key* to a new dictionary with *data* and *size*.

    Raises
    ------
    ValueError
        If *key* is *None* or
        if *key* is already in allocated set or
        if *data* is not an *ndarray*
    """

    if not isinstance(data, ndarray):
        msg = 'Data element is _NOT_ a numpy array'
        raise ValueError(msg)

    if size is None:
        if data is None:
            size = 0
        else:
            size = data.nbytes
    return engine_alloc(conf, 'npy_data', key, data, size)


def get_npy_data(conf, key):
    """Extracts numpy data for variable *key*

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    key : str
        Data entry dict key

    Returns
    -------
    output : ndarray
        Data entry corresponding to *key* in conf['npy_data']
        *None* if *key* is not in conf['npy_data']
    """

    return get_engine_data(conf, 'npy_data', key)


def get_npy_size(conf, key):
    """Extracts numpy data size for variable *key*

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    key : str
        Data entry dict key

    Returns
    -------
    output : int
        Size entry corresponding to *key* in conf['npy_data']
        Zero if *key* is not in conf['npy_data']
    """

    return get_engine_size(conf, 'npy_data', key)


def get_npy_total_size(conf):
    """Extracts the total size of allocated numpy data

    Parameters
    ----------
    conf : dict
        Configuration dictionary

    Returns
    -------
    output : int
        Sum of all size entries in conf['npy_data']
    """

    return get_engine_total_size(conf, 'npy_data')


def npy_free(conf, key, ignore_missing=False):
    """Free numpy data entry *key* from conf['npy_data'] .

    Does not explicitly free the data, but free happens automatically during
    garbage collection when no more references to the ndarray exist.

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    key : str
        Variable to be freed
    ignore_missing : bool, optional
        If *True* the numpy data for variable *key* is freed if present.
        If *False* exceptions may be raised
        based on the value of *key* (see Raises),

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

    return engine_free(conf, 'npy_data', key, ignore_missing)


def npy_free_all(conf, garbage_collect=True):
    """Free all numpy data entries from conf['npy_data'].

    Unreachable allocations from ndarray are automatically freed during
    garbage collect, so there's no need to explicitly free them as long as no
    other references are preserved.

    Parameters
    ----------
    conf : dict
        Configuration dictionary
    garbage_collect : bool, optional
        Flag to enable/disable explicit garbage collection as last action

    Returns
    -------
    output : dict
        Configuration dictionary
    """

    return engine_free_all(conf, 'npy_data', garbage_collect)

