"""
This module converts fisheye image into panoramic image. 
"""

import cv2
import numpy
import numpy.linalg
import math
import find_checkerboard
import math_util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PI = math.pi


def equ_to_vector(x, y, r=1.0):
    """
    This function converts a coordinate in 2D Equirectangular projection to its 3D point. 
    It assume the projection is on a sephere(which it converts back to). 

    Args:
      x: Horizontal coordinate
      y: Vertival coordinate
      r: The radius of the sephere

    Return:
      Px: x of 3D point
      Py: y of 3D point
      Pz: z of 3D point
    """
    lon = x * PI
    lat = y * PI / 2.0

    Px = r * math.cos(lat) * math.cos(lon)
    Py = r * math.cos(lat) * math.sin(lon)
    Pz = r * math.sin(lat)
    return Px, Py, Pz


def vector_to_r_theta(px, py, pz, aperture):
    """
    This function converts a coordinate on a sphere, to r, theta in projection. What we have is an axis, y axis that is
    where our lens is pointing to, and r defines the angle of y axis to the point p which is defined by px, py, pz.
    Theta is the rotation of the point p on y axis. Note: P = (Px, Py, Pz), |P| = 1.

    Args:
      px: x of 3D point
      py: y of 3D point
      pz: z of 3D point

    Return:
      r: r, Angle of PO and y axis
      theta: Rotation of point p on y axis.
    """
    r = 2 * math.atan2(math.sqrt(px*px + pz*pz), py)/aperture
    theta = math.atan2(pz, px)
    return r, theta


def fisheye_coord_to_fi_theta(fish_coord, aperture):
    """
    This function translates fisheye coordinate to fi, theta. Please see: http://paulbourke.net/dome/dualfish2sphere/

    Args:
        fish_coord: 3x1 vector. The values should be normalized, where x = [-1,1], y = [-1,1]
        aperture: The Field of view in radius.

    Returns:
        fi: Fi in radius.
        theta: Theta in radius.

    """
    fi = numpy.linalg.norm(fish_coord) * aperture / 2.0
    theta = math.atan2(fish_coord[1], fish_coord[0])
    return fi, theta


def fi_theta_to_pxyz(fi, theta, o_center_position = numpy.array([0.0,0.0,0.0]), r = 1.0,rotation_matrix = numpy.eye(3)):
    vec = numpy.array([math.sin(fi) * math.cos(theta),\
                        math.cos(fi),\
                        math.sin(theta)*math.sin(fi)])
    vec = rotation_matrix.dot(vec)
    o_center_length = numpy.linalg.norm(o_center_position)
    x1,x2 = math_util.solve_quadratic(1.0, 2.0 * numpy.inner(vec, o_center_position), o_center_length * o_center_length - r * r)
    return x1 * vec + o_center_position


def pxyz_to_equ(p_coord):
    lon = math.atan2(p_coord[1], p_coord[0])
    lat = math.atan2(p_coord[2], numpy.linalg.norm(p_coord[0:2]))
    return lon / math.pi, 2.0 * lat / math.pi


def calculate_error_2D(points_a, points_b, r):
    difference = 0.0
    points_a/= r
    points_b/= r
    i = 0
    for p_a, p_b in zip(points_a, points_b):
        coord_a = numpy.array(pxyz_to_equ(p_a))
        coord_b = numpy.array(pxyz_to_equ(p_b))
        i += 1
        difference += numpy.linalg.norm(coord_a-coord_b)

    return difference / len(points_b)


def print_points_2d(points_a, points_b, r):
    img = numpy.zeros((1024,2048,3),dtype=numpy.uint8)
    points_a/= r
    points_b/= r
    for p_a in points_a:
        coord_a = numpy.array(pxyz_to_equ(p_a))
        coord_a_x = int(math_util.lerp(coord_a[0],-1,1,0,2048))
        coord_a_y = int(math_util.lerp(coord_a[1],-1,1,0,1024))
        img[coord_a_y,coord_a_x,:] = [255,0,0]

    for p_b in points_b:
        coord_b = numpy.array(pxyz_to_equ(p_b))
        coord_b_x = int(math_util.lerp(coord_b[0],-1,1,0,2048))
        coord_b_y = int(math_util.lerp(coord_b[1],-1,1,0,1024))
        img[coord_b_y,coord_b_x,:] = [0,255,0]
    cv2.imwrite("template.png",img)


def dual_fisheye_calibrate(image_l, image_r,fov_l,fov_r,points_a=None, points_b=None):

    # for both image
    image_boards = numpy.array([[[1/2.0,0.0,1/2.0,1/2.0],[1/4.0,0.0,1/4.0,1/4.0],[0.0,1/4.5,1/4.0,1/4.0],[0.0,1/2.0,1/3.0,1/3.0],[1/4.0,3/4.0,1/4.0,1/4.0],[1/2.0,3/4.0,1/4.0,1/4.0]],\
                                [[1/4.0,0.0,1/4.0,1/4.0],[1/2.0,0.0,1/4.0,1/4.0],[3/4.0,1/4.5,1/4.0,1/4.0],[3/4.0,1/2.0,1/3.0,1/3.0],[1/2.0,3/4.0,1/4.0,1/4.0],[1/4.5,3/4.0,1/4.0,1/4.0]]])
    all_points = []
    ite = 100
    
    if points_a and points_b != None:
        all_points = [points_a, points_b]
    else:
        # 1. find points
        for image, boards in zip([image_l,image_r],image_boards):
            h,w = image.shape[:2]
            length = numpy.array(image.shape[:2])

            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            board_points = numpy.zeros((0,2))
            for board in boards:
                mask = numpy.array([board[0]*w,board[1]*h,board[2]*w,board[3]*h], dtype=numpy.int32)
                point = find_checkerboard.find_checkerboard(mask,gray_img)
                point = numpy.squeeze(point, axis = 1)
                print board_points.shape, point.shape
                board_points = numpy.vstack((board_points, point))
            board_points /= length/2.0
            #print numpy.tile([1.0,1.0],(len(board_points),2))
            board_points -= numpy.tile([1.0,1.0],(len(board_points),1))
            all_points.append(board_points)
        

    # 2. unwarp and project to global pxyz
    def find_r_t(fov_l,fov_r,image_center_l=numpy.array([0.0,0.0]),\
                             image_center_r=numpy.array([0.0,0.0])):
        rotation_matrix_l = numpy.eye(3) #math_util.rotation_matrix_z(math.pi/2.0)
        rotation_matrix_r = numpy.eye(3)
        r = 1000.0
        det_y_l = numpy.array([0.0,0.0,0.0])
        det_y_r = numpy.array([0.0,0.0,0.0])
        ite = 50
        eps = 1E-10
        old_error = float("inf")
        error = 1000.0

        while old_error - error > eps:
            l_reproject_pts = reproject_coords(all_points[0]+image_center_l,fov_l, r, det_y_l, rotation_matrix_l)
            r_reproject_pts = reproject_coords(all_points[1]+image_center_r,fov_r, r, det_y_r, rotation_matrix_r)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(l_reproject_pts[:,0], l_reproject_pts[:,1],l_reproject_pts[:,2],c='r')
            # ax.scatter(r_reproject_pts[:,0], r_reproject_pts[:,1], r_reproject_pts[:,2],c='b')
            # plt.show()
            #error =  calculate_error(numpy.eye(3), numpy.zeros((1,3)),l_reproject_pts,r_reproject_pts)
            old_error = error
            error = calculate_error_2D(l_reproject_pts.copy(), r_reproject_pts.copy(), 1000.0)


            # 3. find translation + rotation for points
            new_rotation,t = math_util.find_rigid_transformation(numpy.mat(r_reproject_pts),numpy.mat(l_reproject_pts))

            # 4. calculate error
            #print rotation_matrix_l

            det_y_l -= numpy.array(t)[:,0]/2.0
            det_y_r += numpy.array(t)[:,0]/2.0

            numpy.matmul(new_rotation,rotation_matrix_r,rotation_matrix_r)

            #print calculate_error(new_rotation, t,l_reproject_pts,r_reproject_pts)
        #print_points_2d(l_reproject_pts.copy(), r_reproject_pts.copy(), 1000.0)
        return (error,rotation_matrix_l.T,det_y_l , rotation_matrix_r.T, det_y_r)
    min_value = float("inf")
    min_result = None
    min_fov = None

    for d_fov_l in xrange(-0,1,1):
        for d_fov_r in xrange(-0,1,1):
            # for d_img_center_l_x in xrange(-3,3,1):
            #     for d_img_center_l_y in xrange(-3,3,1):
            #         for d_img_center_r_x in xrange(-3,3,1):
            #             for d_img_center_r_y in xrange(-3,3,1):
            try:
                result = find_r_t(fov_l+d_fov_l/180.0 * math.pi, fov_r+d_fov_r/180.0 * math.pi,\
                )#numpy.array([d_img_center_l_x,d_img_center_l_y])/3000.0,numpy.array([d_img_center_r_x,d_img_center_r_y])/3000.0
            except:
                raise
            if result[0] < min_value:
                min_value = result[0]
                min_result = result
                min_fov = (fov_l+d_fov_l/180.0 * math.pi, fov_r+d_fov_r/180.0 * math.pi)

    return min_result,min_fov



def calculate_error(r,t,pts_a,pts_b):
    """
    This function calculates error by measuring the distance between 
    corresponding r(p_a) + t and p_b

    Args:
      r: 3x3 numpy rotation matrix
      t: 3x1 numpy translation array
      pts_a: nx3 3D points
      pts_b: nx3 3D points

    Return:
      diff: Average difference of transformed a and b points
    """
    diff = 0.0
    for p_a, p_b in zip(pts_a,pts_b):
        diff += numpy.linalg.norm(r.dot(p_a)+t-p_b)
    return diff/len(p_a)

def reproject_coords(coords, fov, r, o_center_position, rotation_matrix):
    result = numpy.zeros((len(coords), 3))
    for i,fish_coord in enumerate(coords):
        fi,theta = fisheye_coord_to_fi_theta(fish_coord,fov)
        # print fi,theta
        result[i] = fi_theta_to_pxyz(fi,theta,o_center_position,r,rotation_matrix)
    return result

def vector_to_r_theta_with_delta_y(px,py,pz, aperture, dy, rotation_matrix):
    vec = numpy.array([px,py,pz])-dy
    vec = vec/numpy.linalg.norm(vec)
    px,py,pz = rotation_matrix.dot(vec)

    r = 2 * math.atan2(math.sqrt(px*px + pz*pz),py)/aperture
    theta = math.atan2(pz,px)
    return r, theta


def vector_to_fisheye(px,py,pz, aperture, det_y, rotation_matrix = numpy.eye(3)):
    """
    This function takes a 3D point on unit sephere, and map it to 2D fish eye
    coordinate.

    Args:
      px: x of 3D point
      py: y of 3D point
      pz: z of 3D point
      aperture: Field of view of the image in radius.
      det_y: The difference between lens principal point and origin point(0,0,0), we only allow 1D adjust.

    Return:
      x: x of 2D point on normalized plane
      y: y of 2D point on normalized plane
    """
    r,theta = vector_to_r_theta_with_delta_y(px,py,pz,aperture, det_y,rotation_matrix)

    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x,y


def generate_map(input_image_size, output_image_size, aperture, rotation_matrix, o_center_position):
    """
    This function generates a map for openCV's remap function.

    Args:
      input_image_size: Tuple of the input image size, (y,x)
      output_image_size: Tuple of the output image size, (y,x)
      aperture: Field of view of the image in radius.
      rotation_matrix: Not used for now

    Return:
      Matrix: A matrix same size as output_image.
    """
    print "input_image_size", input_image_size
    print "output_image_size", output_image_size
    if output_image_size[0] * 2 != output_image_size[1]:
        print "error output_image_size"
    else:
        image_map = numpy.zeros((output_image_size[0], output_image_size[1], 2), dtype= numpy.float32)
        for x in xrange(output_image_size[1]):
            for y in xrange(output_image_size[0]):
                normal_x = math_util.lerp(x,0,output_image_size[1] - 1,-1,1)
                normal_y = - math_util.lerp(y,0,output_image_size[0] - 1,-1,1)#invert y value

                px,py,pz = equ_to_vector(normal_x, normal_y, r = 1000.0)
                normal_fish_x, normal_fish_y = vector_to_fisheye(px,py,pz,aperture, o_center_position,rotation_matrix)

                fish_x = math_util.lerp(normal_fish_x, -1,1, 0, input_image_size[1] -1)
                fish_y = math_util.lerp(normal_fish_y, -1,1, 0, input_image_size[0] -1)

                image_map[y,x] = fish_x, fish_y
        return image_map


def convert_fisheye_equ(input_image, output_image_size, aperture, rotation_matrix,o_center_position):
    """
    This function convert a equidistant projected fisheye lens into an equirectangular projection image.

    Args:
      input_image: input image should be numpy matrix
      output_image_size: Tuple of the output image size, (y,x)
      aperture: Fild of view of the image in radius.
      rotation_matrix: Not used for now

    Return:
      matrix: unwarped image.
    """
    image = numpy.zeros((output_image_size[1],output_image_size[0],3),dtype = numpy.uint8)
    image_map = generate_map(input_image.shape, image.shape, aperture, rotation_matrix,o_center_position)

    cv2.remap(src=input_image, dst=image, map1=image_map, map2=None,interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(output_image_name,image)
    # cv2.imshow("aaa",image)
    # c = cv2.waitKey(0)
    return image


def rotation():
    for i in xrange(-8,8):
        r = math.pi / 4.0 * i
        r_m = rotation_matrix_z(r)
        print r_m
        input_image_l = cv2.imread("fisheye_l.jpg")
        output_image = convert_fisheye_equ(input_image_l,"out_l.jpg", (1024,512), 200.0/180*math.pi, r_m,numpy.array([0.0,0.0,0.0]))
        cv2.imwrite("rotation_test_{}.jpg".format(i*45), output_image)


if __name__ == "__main__":
    input_image_r = cv2.imread("image/fisheye_r.jpg")
    input_image_l = cv2.imread("image/fisheye_l.jpg")

    # rotation()
   

    result, fov = dual_fisheye_calibrate(input_image_l,input_image_r,200 /180.0 *math.pi,200 /180.0 *math.pi)
    #print fov[0]/ math.pi*180,fov[1]/math.pi*180
    error, r_l,t_l, r_r,t_r = result
    print "Reproject Error:",error
    print "Left image rotation matrix:"
    print r_l
    print "Left image Translation:"
    print t_l
    print "Right image rotation matrix:"
    print r_r
    print "Right image Translation"
    print t_r

    # e = numpy.array(math_util.rotation_matrix_decompose(r_r))
    # print e / math.pi*180


    output_image_l = convert_fisheye_equ(input_image_l, (1024,512), fov[0], r_l, t_l)
    output_image_r = convert_fisheye_equ(input_image_r, (1024,512), fov[1], r_r, t_r)
    cv2.imwrite("out_l_1024.jpg",output_image_l)
    cv2.imwrite("out_r_1024.jpg",output_image_r)



