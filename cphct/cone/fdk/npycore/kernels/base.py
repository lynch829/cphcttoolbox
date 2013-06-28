#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - numpy specific FDK reconstruction kernels
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

"""Step and shoot cone beam CT kernels using the FDK algorithm"""

from cphct.npycore import radians, cos, sin, arctan, dot, divide, \
    zeros, ones, array, hstack, vstack, fft, real, rint, int32, \
    meshgrid, sqrt, allowed_data_types
from cphct.npycore.io import get_npy_data, save_auto
from cphct.npycore.utils import log_checksum
from cphct.npycore.misc import linear_coordinates

from cphct.log import logging
from cphct.misc import timelog


# These are basic numpy functions exposed through numpyext to use same numpy

def __rotation_matrix(
    phi,
    theta,
    psi,
    fdt,
    ):
    """Generate a three dimensional rotation matrix

    References:
    http://www.fastgraph.com/makegames/3drotation/
    http://en.wikipedia.org/wiki/3D_projection#Perspective_projection

    Parameters
    ----------
    phi : float
       x-axis rotation in radians
    theta : float
       y-axis rotation in radians
    psi : float
       z-axis rotation in radians
    fdt : dtype
       Float data type (internal precision)
    """

    rotate_phi = array([(1, 0, 0), (0, cos(phi), sin(phi)), (0,
                       -sin(phi), cos(phi))], dtype=fdt)

    rotate_theta = array([(cos(theta), 0, -sin(theta)), (0, 1, 0),
                         (sin(theta), 0, cos(theta))], dtype=fdt)

    rotate_psi = array([(cos(psi), sin(psi), 0), (-sin(psi), cos(psi),
                       0), (0, 0, 1)], dtype=fdt)

    rotate = dot(rotate_phi, dot(rotate_theta, rotate_psi))

    return rotate


def generate_transform_matrix(
    proj_angle_rad,
    detector_pixel_width,
    detector_pixel_height,
    detector_column_shift,
    detector_row_shift,
    source_distance,
    detector_distance,
    detector_shape,
    fdt,
    ):
    """Generates a transformation matrix used to project 3D voxel coordinates
    to pixel coordinates in the 2D detector plane

    References:
    http://en.wikipedia.org/wiki/3D_projection#Perspective_projection
    http://www.fastgraph.com/makegames/3drotation/
    http://knol.google.com/k/matrices-for-3d-applications-translation-rotation
    http://www.falloutsoftware.com/tutorials/gl/gl0.htm

    Parameters
    ----------
    proj_angle_rad : float
       Projection angle in radians
    detector_pixel_width : float
       Detector pixel width in cm
    detector_pixel_height : float
       Detector pixel height in cm
    detector_column_shift : float
       Center ray aligned pixel column shift
    detector_row_shift : float
       Center ray aligned pixel row shift
    source_distance : float
       Distance in cm from x-ray source to iso center
    detector_distance : float
       Distance in cm from isocenter to detector
    detector_shape : str
       Shape of the detector
    fdt : dtype
       Float data type (internal precision)

    Returns
    -------
    output : ndarray
        A transformation matrix used to project 3D voxel coordinates
        to pixel coordinates in the 2D detector plane
    """

    float_cutoff = 1e-10

    rad_90 = radians(90.0)
    source_detector_distance = source_distance + detector_distance

    # Using a flat detector shape, detector alignment and the transformation
    # from centimeters to pixel values is a part of the transformation matrix
    #
    # For the curved detector this is done explicitly
    # in the reconstruction process

    if detector_shape == 'flat':
        inv_detector_pixel_width = 1.0 / detector_pixel_width
        inv_detector_pixel_height = 1.0 / detector_pixel_height
        pixel_row_shift = detector_row_shift
        pixel_column_shift = detector_column_shift
    else:
        inv_detector_pixel_width = 1.0
        inv_detector_pixel_height = 1.0
        pixel_row_shift = 0
        pixel_column_shift = 0

    # The rotation matrix is used to align the xray source and the detector
    # in the voxel coordinate system

    rotation = __rotation_matrix(rad_90, 0.0, proj_angle_rad + rad_90,
                                 fdt)

    cutoff_indexes = abs(rotation) < float_cutoff
    rotation[cutoff_indexes] = 0.0

    # Calculate the xray source position

    xray_source = array([[source_distance * cos(proj_angle_rad)],
                        [source_distance * sin(proj_angle_rad)], [0]],
                        dtype=fdt)

    # Transform the xray source position to voxel coordinates

    rotated_xray_source = dot(-rotation, xray_source)

    # Create detector panel matrix

    det_panel = array([(-source_detector_distance, 0, 0, 0), (0,
                      -source_detector_distance, 0, 0), (0, 0, 1, 0)],
                      dtype=fdt)

    # Align xray source and detector panel to the voxel coordinate system.
    # This generates the basis for the transform matrix used to map 3D voxels
    # onto 2D detector pixels

    transform_matrix = dot(det_panel, hstack([vstack([rotation,
                           zeros((1, 3), dtype=fdt)]),
                           vstack([rotated_xray_source, ones((1, 1),
                           dtype=fdt)])]))

    # Align voxel iso-center with detector center
    # and transform coordinates from centimeters to pixels
    # NOTE: flat detector shape only (see above)

    pixel_remap = array([(inv_detector_pixel_width, 0,
                        pixel_column_shift), (0,
                        inv_detector_pixel_height, pixel_row_shift),
                        (0, 0, 1)], dtype=fdt)

    transform_matrix = dot(pixel_remap, transform_matrix)

    # Move view plane from xray source position into iso-center (voxel: 0,0,0)

    transform_matrix /= -source_distance

    # Cut off values less than 1e-10 as they are considered noise

    cutoff_indexes = abs(transform_matrix) < float_cutoff
    transform_matrix[cutoff_indexes] = 0.0

    return transform_matrix


def generate_combined_matrix(
    x_min,
    x_max,
    x_voxels,
    y_min,
    y_max,
    y_voxels,
    fdt,
    ):
    """
    Generates a matrix containing the coordinates of all
    x and y voxel combinations for a single z voxel coordinate.
    Additional space is allocated for a dummy dimension needed
    by dot products performed during reconstruction.

    Parameters
    ----------
    x_min : float
       Field of View minimum x coordinate in cm.
    x_max : float
       Field of View maximum x coordinate in cm.
    x_voxels : int
       Field of View resolution in x.
    y_min : float
       Field of View minimum y coordinate in cm.
    y_max : float
       Field of View maximum y coordinate in cm.
    y_voxels : int
       Field of View resolution in y.
    fdt : type
        Output filter data type.

    Returns
    -------
    output : ndarray
        Returns a combination of all *x_voxels* by *y_voxels* coordinate
        positions with dtype *fdt*.
    """

    x_coords = linear_coordinates(x_min, x_max, x_voxels, True, fdt)
    y_coords = linear_coordinates(y_min, y_max, y_voxels, True, fdt)

    (y_coords_grid, x_coords_grid) = meshgrid(y_coords, x_coords)
    (flat_y_coords_grid, flat_x_coords_grid) = \
        (fdt(y_coords_grid.flatten('F')), fdt(x_coords_grid.flatten('F'
         )))

    # Allocate space for one slice, the z values are filled in
    # when looping over slices in the reconstruction kernel

    flat_z_coords_grid = zeros(len(flat_x_coords_grid), dtype=fdt)

    # Dummy must be ones in order to get correct dot products
    # in the reconstruction kernel

    dummy = ones(len(flat_x_coords_grid), dtype=fdt)

    combined_matrix = vstack([flat_x_coords_grid, flat_y_coords_grid,
                             flat_z_coords_grid, dummy])

    return combined_matrix


def generate_detector_boundingboxes(conf, fdt):
    """
    Generate projection bounding boxes for each reconstruction chunk

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    fdt : type
       Output filter data type.

    Returns
    -------
    output : ndarray
        Returns a ndarray of shape (nr_chunks, 2, 2) with projection
        [start_row,end_row*][*start_column,end_column] for each chunk
        
    """

    detector_rows = conf['detector_rows']
    detector_columns = conf['detector_columns']
    detector_pixel_width = conf['detector_pixel_width']
    detector_pixel_height = conf['detector_pixel_height']
    detector_column_shift = conf['detector_column_shift']
    detector_row_shift = conf['detector_row_shift']
    source_distance = conf['source_distance']
    detector_distance = conf['detector_distance']
    detector_shape = conf['detector_shape']
    x_min = conf['x_min']
    x_max = conf['x_max']
    y_min = conf['y_min']
    y_max = conf['y_max']
    z_min = conf['z_min']
    z_max = conf['z_max']
    z_voxels = conf['z_voxels']
    chunk_size = conf['chunk_size']

    # Use Pythagoras to calculate the farest corner of FoV
    # We need to find the largest span of FoV which defines
    # the largest cone-angle

    maximum_x = max(abs(x_min), abs(x_max))
    maximum_y = max(abs(y_min), abs(y_max))
    fov_diagonal_radius = sqrt(maximum_x ** 2 + maximum_y ** 2)

    # Use 3D projection: http://en.wikipedia.org/wiki/3D_projection
    # to calculate the minimum and maximum row posistions for each chunk

    # Generate the transform matrix used to project 3D voxels to 2D pixels
    # We need to find the largest cone angle for each chunk.

    # Using the FoV farest corner voxels with the found diagonal:
    # (FoV_diagonal_radius, 0, Z_0) and (-FoV_diagonal_radius, 0, Z_1)
    # provides the maximum cone angle for the chunk starting at Z_0 ending in Z_1.

    proj_angle_rad = 0.0

    transform_matrix = generate_transform_matrix(
        proj_angle_rad,
        detector_pixel_width,
        detector_pixel_height,
        detector_column_shift,
        detector_row_shift,
        source_distance,
        detector_distance,
        detector_shape,
        fdt,
        )

    voxel_coordinate = vstack([[fov_diagonal_radius,
                              -fov_diagonal_radius], [0, 0], [0, 0],
                              [1, 1]])

    chunks = z_voxels / chunk_size
    chunk_size_cm = (z_max - z_min) / z_voxels * chunk_size

    detector_boundingboxes = zeros((chunks, 2, 2),
                                   dtype=allowed_data_types['uint32'])
    detector_boundingboxes[:, 1, 0] = 0
    detector_boundingboxes[:, 1, 1] = detector_columns

    z_last = z_min

    for chunk in xrange(chunks):
        z_first = z_last
        voxel_coordinate[2] = z_first

        (map_first_rows, _) = project_voxels_to_pixels(conf,
                transform_matrix, voxel_coordinate)

        first_row = map_first_rows.min()
        if first_row < 0:
            first_row = 0

        z_last = z_first + chunk_size_cm
        voxel_coordinate[2] = z_last
        (map_last_rows, _) = project_voxels_to_pixels(conf,
                transform_matrix, voxel_coordinate)

        last_row = map_last_rows.max()
        if last_row < 0:
            last_row = 0
        elif last_row >= detector_rows:
            last_row = detector_rows - 1

        detector_boundingboxes[chunk, 0, 0] = first_row
        detector_boundingboxes[chunk, 0, 1] = last_row + 1

    return detector_boundingboxes


def generate_proj_weight_matrix(
    detector_rows,
    detector_columns,
    detector_row_shift,
    detector_column_shift,
    detector_pixel_height,
    detector_pixel_width,
    source_distance,
    detector_distance,
    detector_shape,
    fdt,
    ):
    """Calculate the cone-beam flat or curved projection weight

    Parameters
    ----------
    detector_rows : int
       Number of pixel rows in projections
    detector_columns : int
       Number of pixel columns in projections
    detector_row_shift : float
       Center ray aligned pixel row shift
    detector_column_shift : float
       Center ray aligned pixel column shift
    detector_pixel_height : float
       Detector pixel height in cm
    detector_pixel_width : float
       Detector pixel width in cm
    source_distance : float
       Distance in cm from source to isocenter
    detector_distance : float
       Distance in cm from isocenter to detector
    detector_shape : str
       Shape of detector
    fdt : dtype
       Output proj_weight data type.

    Returns
    -------
    output : ndarray
        Returns a proj_weight matrix of *detector_rows* by *detector_columns*
        with dtype *fdt*.
    """

    # From Henrik Turbell's Ph.D thesis:
    #    link: http://www2.cvl.isy.liu.se/ScOut/Theses/PaperInfo/turbell01.html
    #    link: http://www2.cvl.isy.liu.se/Research/Tomo/Turbell/abstract.html
    #
    #    Bibtex:
    #       @PhdThesis{turbell01,
    #       author  = {Henrik Turbell},
    #       title   = {Cone-Beam Reconstruction Using Filtered Backprojection},
    #       school  = {Link{\"o}ping University, Sweden},
    #       year    = {2001},
    #       month   = {February},
    #       address = {SE-581 83 Link\"oping, Sweden},
    #       node    = {Dissertation No. 672, ISBN 91-7219-919-9}
    #       }

    source_detector_distance = source_distance + detector_distance

    # Detector pixel center coordinates in both directions, using coordinate
    # system centered in the middle of the detector.

    cols = (-detector_column_shift
            + linear_coordinates(-detector_columns / 2.0,
            detector_columns / 2.0, detector_columns, True, fdt)) \
        * detector_pixel_width

    rows = (-detector_row_shift + linear_coordinates(-detector_rows
            / 2.0, detector_rows / 2.0, detector_rows, True, fdt)) \
        * detector_pixel_height

    if detector_shape == 'curved':

        # Part of equation 3.9 Turbell

        col_rads = abs(cols / source_detector_distance)
        col_weight = cos(col_rads)

        (col_weight_grid, row_grid) = meshgrid(col_weight, rows)

        row_weight_grid = source_detector_distance \
            / sqrt(source_detector_distance ** 2 + row_grid ** 2)

        proj_weight_matrix = col_weight_grid * row_weight_grid
    else:

        # Equation 3.5 Turbell

        (col_grid, row_grid) = meshgrid(cols, rows)
        proj_weight_matrix = source_detector_distance \
            / sqrt(source_detector_distance ** 2 + row_grid ** 2
                   + col_grid ** 2)

    return proj_weight_matrix


def project_voxels_to_pixels(conf, transform_matrix, combined_matrix):
    """Project 3D voxel coordinates to 2D pixel indexes
    
    Parameter
    ---------
    conf : dict
        A dictionary of configuration options.
    transform_matrix : ndarray
       Matrix used to project 3D voxel coordinates
       to pixel coordinates in the 2D detector plane
    combined_matrix : ndarray
       Matrix containing the combination of all voxel coordinates
       in the x and y plane.
    
    Returns
    -------
    output : tuple
       Tuple of (row, col) ndarrays with pixel coordinates
    """

    # Find the mapping between volume voxels and detector pixels

    vol_det_map = dot(transform_matrix, combined_matrix)

    flat_map_cols = divide(vol_det_map[0, :], vol_det_map[2, :])
    flat_map_rows = divide(vol_det_map[1, :], vol_det_map[2, :])

    if conf['detector_shape'] == 'curved':
        source_distance = conf['source_distance']
        detector_pixel_height = conf['detector_pixel_height']
        detector_pixel_width = conf['detector_pixel_width']
        detector_row_shift = conf['detector_row_shift']
        detector_column_shift = conf['detector_column_shift']

        # Find the xray angle between each detector column pixel
        #
        # By definition of tangens:
        # ---------------------
        # flat_map_cols = source_distance * tan(alpha)
        # tan(alpha) = flat_map_cols/source_distance
        # alpha = tan-1(flat_map_cols/source_distance)

        alpha = arctan(divide(flat_map_cols, source_distance))

        # By definition of an arc:
        # http://en.wikipedia.org/wiki/Arc_%28geometry%29

        curved_map_cols = source_distance * alpha

        # Transform units from cm to pixels

        curved_map_cols *= 1.0 / detector_pixel_width

        # Offset to center in column direction

        curved_map_cols += detector_column_shift

        # From the paper:
        # "Exact helical reconstruction using native cone-beam geometries"
        # http://citeseerx.ist.psu.edu/viewdoc/download?\
        #        doi=10.1.1.121.6008&rep=rep1&type=pdf

        curved_map_rows = flat_map_rows * cos(alpha)

        # Transform units from cm to pixels

        curved_map_rows *= 1.0 / detector_pixel_height

        # Offset to center in row direction

        curved_map_rows += detector_row_shift

        # Round and cast to int

        map_cols_tmp = rint(curved_map_cols)
        map_rows_tmp = rint(curved_map_rows)
    else:
        map_cols_tmp = rint(flat_map_cols)
        map_rows_tmp = rint(flat_map_rows)

    # rint returns float, convert to int

    map_rows = map_rows_tmp.astype(int32)
    map_cols = map_cols_tmp.astype(int32)

    return (map_rows, map_cols)


def generate_volume_weight(
    proj_angle_rad,
    combined_matrix,
    source_distance,
    detector_shape,
    volume_weight_factor,
    ):
    """
    Generates a volume weight matrix used to compensate for
    the distance between the xray source and each voxels in the
    reconstructed volume.

    Parameter
    ---------
    combined_matrix : ndarray
       Matrix containing the combination of all voxel coordinates
       in the x and y plane.
    source_distance : float
       Distance in cm from source to isocenter.
    detector_shape : str
       Shape of detector.
    volume_weight_factor : float
       The fraction of contribution from each projection

    Returns
    -------
    output : ndarray
       A volume weight matrix used to compensate for
       the distance between the xray source and each voxels in the
       reconstructed volume.
    """

    # Theory from Henrik Turbell's Ph.D thesis:
    # link: http://www2.cvl.isy.liu.se/ScOut/Theses/PaperInfo/turbell01.html
    # link: http://www2.cvl.isy.liu.se/Research/Tomo/Turbell/abstract.html
    #
    # Bibtex:
    #
    # @PhdThesis{turbell01,
    # author  = {Henrik Turbell},
    # title   = {Cone-Beam Reconstruction Using Filtered Backprojection},
    # school  = {Link{\"o}ping University, Sweden},
    # year    = {2001},
    # month   = {February},
    # address = {SE-581 83 Link\"oping, Sweden},
    # node    = {Dissertation No. 672, ISBN 91-7219-919-9}
    # }

    cos_angle = cos(proj_angle_rad)
    sin_angle = sin(proj_angle_rad)

    if detector_shape == 'curved':

        # D = source_distance
        # L = source_voxel_dist
        # b = projection angle
        #
        # Turbell (2.23) page 18
        # L(x,y,b) = sqrt((R + x cos b + y sin b)**2 + (-x sin b + y cos b)**2)
        #
        # We have an inverted z axis due to source-camera geometry,
        # see geometry document
        #
        # L(x, y, b) = sqrt((R - (x cos b + y sin b))**2
        #              + (-x sin b + y cos b)**2)
        #
        # volume_weight = D^2/L^2
        #
        # L^2 = sqrt((R - (x cos b + y sin b))**2 + (-x sin b + y cos b) **2) *
        #       sqrt((R - (x cos b + y sin b))**2 + (-x sin b + y cos b) **2)
        #     = (R - (x cos b + y sin b))**2 + (-x sin b + y cos b) **2
        #
        # volume_weight = D^2/L^2

        L_2 = (source_distance - (cos_angle * combined_matrix[0]
               + sin_angle * combined_matrix[1])) ** 2 + (-sin_angle
                * combined_matrix[0] + cos_angle * combined_matrix[1]) \
            ** 2

        volume_weight = source_distance ** 2 / L_2 \
            * volume_weight_factor
    else:

        # Turbell (2.29) page 19
        # U (x, y, b) = R + x cos b + y sin b
        #
        # We have an inverted z axis due to source-camera geometry,
        # see geometry document
        #
        # U (x, y, b) = R - (x cos b + y sin b)

        U = source_distance - (cos_angle * combined_matrix[0]
                               + sin_angle * combined_matrix[1])
        volume_weight = source_distance ** 2 / U ** 2 \
            * volume_weight_factor

    return volume_weight


def reconstruct_proj(
    conf,
    proj,
    z_voxels_array,
    fdt,
    ):
    """Reconstructs a single projection
    conf : dict
        A dictionary of configuration options.
    proj : dict
        Projection dictionary containing meta infomation and data
    z_voxels_array : ndarray
        Array of all Z voxel positions
    fdt : dtype
        Float precision

    Returns
    -------
    output : dict
        The dictionary of configuration options.
    """

    detector_columns = conf['detector_columns']
    detector_rows = conf['detector_rows']
    detector_pixel_width = conf['detector_pixel_width']
    detector_pixel_height = conf['detector_pixel_height']
    detector_column_shift = conf['detector_column_shift']
    detector_row_shift = conf['detector_row_shift']
    source_distance = conf['source_distance']
    detector_distance = conf['detector_distance']
    detector_shape = conf['detector_shape']
    proj_filter_width = conf['proj_filter_width']

    proj_weight_matrix = get_npy_data(conf, 'proj_weight_matrix')
    combined_matrix = get_npy_data(conf, 'combined_matrix')
    proj_filter_matrix = get_npy_data(conf, 'proj_filter_matrix')
    recon_chunk = get_npy_data(conf, 'recon_chunk')

    proj_angle_rad = radians(fdt(proj['angle']))
    proj_index = conf['app_state']['backproject']['proj_idx']
    proj_data_index = proj_index - conf['app_state']['projs']['first']
    proj_data = get_npy_data(conf, 'projs_data')[proj_data_index]

    if not proj['filtered']:

        # proj_weight projection

        if conf['proj_weight'] != 'skip':
            timelog.set(conf, 'verbose', 'proj_weight')
            proj_data *= proj_weight_matrix
            timelog.log(conf, 'verbose', 'proj_weight')

        if conf['checksum']:
            proj_view = proj_data.ravel()
            log_checksum('Weighted projs chunk', proj_view,
                         proj_view.size)

        # If filter is set, apply filter to weigthed projection

        if conf['proj_filter'] != 'skip':
            timelog.set(conf, 'verbose', 'proj_filter')
            filtered_proj_freq = fft.fft(proj_data, proj_filter_width,
                    1) * proj_filter_matrix

            proj_data = fdt(real(fft.ifft(filtered_proj_freq,
                            proj_filter_width, 1)))[:, :
                    detector_columns]

            timelog.log(conf, 'verbose', 'proj_filter')

        if conf['checksum']:
            proj_view = proj_data.ravel()
            log_checksum('Filtered projs chunk', proj_view,
                         proj_view.size)

    # Save filtered projection if requested

    if conf['save_filtered_projs_data_path']:
        logging.debug('Saving filtered projection data')

        timelog.set(conf, 'verbose', 'proj_save')
        fd = open(conf['save_filtered_projs_data_path'], 'r+b', 0)
        fd.seek(fdt(0).nbytes * proj['index'] * proj_data.shape[0]
                * proj_data.shape[1])
        save_auto(fd, proj_data)
        fd.close()
        timelog.log(conf, 'verbose', 'proj_save')

    # Numpy FDK operates on flat arrays

    flat_proj_data = proj_data.ravel()

    # Generate tranform matrix based on projection angle

    timelog.set(conf, 'verbose', 'transform_matrix')
    transform_matrix = generate_transform_matrix(
        proj_angle_rad,
        detector_pixel_width,
        detector_pixel_height,
        detector_column_shift,
        detector_row_shift,
        source_distance,
        detector_distance,
        detector_shape,
        fdt,
        )
    timelog.log(conf, 'verbose', 'transform_matrix')

    # If autogenerate volume z slice weight if not provided by conf

    volume_weight = fdt(1.0)

    if conf['volume_weight'] != 'skip':
        timelog.set(conf, 'verbose', 'volume_weight')
        if conf['volume_weight']:
            volume_weight = get_npy_data(conf, 'volume_weight_matrix'
                    )[proj_index].ravel()
        else:
            volume_weight = generate_volume_weight(proj_angle_rad,
                    combined_matrix, conf['source_distance'],
                    conf['detector_shape'], conf['volume_weight_factor'
                    ])
        timelog.log(conf, 'verbose', 'volume_weight')

    # Reconstruct z slices in the xy plane one at a time

    timelog.set(conf, 'verbose', 'backproject')

    for z in xrange(len(z_voxels_array)):

        # Put current z voxel into combined_matrix

        combined_matrix[2, :] = z_voxels_array[z]

        # Find the detector pixels that contribute to the current slice
        # xrays that hit outside the detector area are masked out

        (map_rows, map_cols) = project_voxels_to_pixels(conf,
                transform_matrix, combined_matrix)

        mask = (map_cols >= 0) & (map_rows >= 0) & (map_cols
                < detector_columns) & (map_rows < detector_rows)

        # The projection pixels that contribute to the current slice

        proj_indexs = map_cols * mask + map_rows * mask \
            * detector_columns

        # Add the weighted projection pixel values to their
        # corresponding voxels in the z slice

        recon_chunk[z].flat += flat_proj_data[proj_indexs] \
            * volume_weight * mask

    timelog.log(conf, 'verbose', 'backproject')

    return conf


