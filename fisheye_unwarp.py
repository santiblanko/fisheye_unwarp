"""
This module converts fisheye image into panoramic video. 
"""

import cv2
import numpy
import numpy.linalg
import math
import find_checkerboard
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



PI = math.pi

def lerp(x, x0,x1,y0,y1):
    """
    This function is a helper function to normalize values.
    Mathmatically, this function does a linear interpolation for x from 
    range [x0,x1] to [y0,y1]. 
    see https://en.wikipedia.org/wiki/Linear_interpolation

    Args:
      x: Value to be interpolated
      x0: Lower range of the original range.
      x1: Higher range of the original range.
      y0: Lower range of the targeted range.
      y1: Higher range of the targeted range.

    Returns:
      float: interpolated value.
    """
    x, x0,x1,y0,y1 = map(float, (x, x0,x1,y0,y1))
    return y0+(x-x0)*(y1-y0)/(x1-x0)

def solve_quadratic(a,b,c):
    if b*b - 4 * a * c < 0:
        return None# no real solution
    else:
        x1 = (-b + math.sqrt(b*b-4 * a * c))/ (2 * a)
        x2 = (-b - math.sqrt(b*b-4 * a * c))/ (2 * a)
        return x1,x2

def rotation_matrix_x(theta):
    return numpy.array([
        [1,0,0],
        [0,math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ])


def equ_to_vector(x,y,r = 1.0):
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
    return Px,Py,Pz

def vector_to_r_theta(px,py,pz, aperture):
    r = 2 * math.atan2(math.sqrt(px*px + pz*pz),py)/aperture
    theta = math.atan2(pz,px)
    return r,theta

def fisheye_coord_to_fi_theta(fish_coord,aperture):
    fi = numpy.linalg.norm(fish_coord) * aperture / 2.0
    theta = math.atan2(fish_coord[1],fish_coord[0])
    return fi,theta

def fi_theta_to_pxyz(fi, theta, o_center_position = numpy.array([0.0,0.0,0.0]), r = 1.0,rotation_matrix = numpy.eye(3)):
    vec = numpy.array([math.sin(fi) * math.cos(theta),\
                        math.cos(fi),\
                        math.sin(theta)*math.sin(fi)])
    vec = rotation_matrix.dot(vec)
    o_center_length = numpy.linalg.norm(o_center_position)
    x1,x2 = solve_quadratic(1.0, 2.0 * numpy.inner(vec, o_center_position), o_center_length * o_center_length - r * r)
    return x1 * vec + o_center_position

def pxyz_to_equ(p_coord):
    lon = math.atan2(p_coord[1],p_coord[0])
    lat = math.atan2(p_coord[2],numpy.linalg.norm(p_coord[0:2]))
    return lon / math.pi, 2.0 * lat / math.pi

def test_fisheye_to_equ():
    import random
    eps = 0.0000001
    rotation_matrix = rotation_matrix_x(60.0 / 180.0 * math.pi)
    r = 1.1
    det_y = random.random()
    fov = 200 /180.0 *math.pi
    for i in xrange(100):
        
        fish_coord = numpy.array([(random.random() - 0.5) * 2.0,(random.random() - 0.5) * 2.0])
        fi,theta = fisheye_coord_to_fi_theta(fish_coord,fov)
        p_coord = fi_theta_to_pxyz(fi,theta,numpy.array([0.0,det_y,0.0]),r,rotation_matrix)
        # print p_coord
        x,y = pxyz_to_equ(p_coord)

        px,py,pz = equ_to_vector(x,y,r)
        # print px,py,pz
        normal_fish_x, normal_fish_y = vector_to_fisheye(px,py,pz,fov, det_y,rotation_matrix.T)
        if abs( fish_coord[0] - normal_fish_x) > eps or abs(fish_coord[1] - normal_fish_y) > eps:
            print normal_fish_x, normal_fish_y
            print fish_coord
            print 'error'



def dual_fisheye_calibrate(image_l, image_r,fov_l,fov_r):

    # for both image
    #[1/2.0,0.0,1/2.0,1/2.0],[1/4.0,0.0,1/4.0,1/4.0],[1/4.0,3/4.0,1/4.0,1/4.0],[1/2.0,3/4.0,1/4.0,1/4.0]
    #[1/4.0,0.0,1/4.0,1/4.0],[1/2.0,0.0,1/4.0,1/4.0],[1/2.0,3/4.0,1/4.0,1/4.0],[1/4.5,3/4.0,1/4.0,1/4.0]
    image_boards = numpy.array([[[1/2.0,0.0,1/2.0,1/2.0],[1/4.0,0.0,1/4.0,1/4.0],[1/4.0,3/4.0,1/4.0,1/4.0],[1/2.0,3/4.0,1/4.0,1/4.0]],\
                                [[1/4.0,0.0,1/4.0,1/4.0],[1/2.0,0.0,1/4.0,1/4.0],[1/2.0,3/4.0,1/4.0,1/4.0],[1/4.5,3/4.0,1/4.0,1/4.0]]])
    all_points = []
    ite = 100

    
    # 1. find points
    for image, boards in zip([image_l,image_r],image_boards):
        h,w = image.shape[:2]
        length = max(w,h)

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        board_points = numpy.zeros((0,2))
        for board in boards:
            mask = numpy.array([board[0]*w,board[1]*h,board[2]*w,board[3]*h], dtype=numpy.int32)
            point = find_checkerboard.find_checkerboard(mask,gray_img)
            point = numpy.squeeze(point, axis = 1)
            #print board_points.shape, point.shape,numpy.tile([1.0,1.0],(len(board_points),2)).shape
            board_points = numpy.vstack((board_points, point))
        board_points /= length/2.0
        #print numpy.tile([1.0,1.0],(len(board_points),2))
        board_points -= numpy.tile([1.0,1.0],(len(board_points),1))
        all_points.append(board_points)

    # 2. unwarp and project to global pxyz
    def find_r_t(fov_l,fov_r):
        rotation_matrix_l = numpy.eye(3)
        rotation_matrix_r = numpy.eye(3)
        r = 1000
        det_y_l = numpy.array([0.0,0.0,0.0])
        det_y_r = numpy.array([0.0,0.0,0.0])
        ite = 10
        while ite > 0:
            l_reproject_pts = reproject_coords(all_points[0],fov_l, r, det_y_l, rotation_matrix_l)
            r_reproject_pts = reproject_coords(all_points[1],fov_r, r, det_y_r, rotation_matrix_r)
            error =  calculate_error(numpy.eye(3), numpy.zeros((1,3)),l_reproject_pts,r_reproject_pts)

           

            # 3. find translation + rotation for points
            new_rotation,t = find_rigid_transformation(numpy.mat(l_reproject_pts),numpy.mat(r_reproject_pts))

            # 4. calculate error
            #print rotation_matrix_l

            det_y_l += numpy.array(t)[:,0]/2.0
            det_y_r -= numpy.array(t)[:,0]/2.0

            numpy.matmul(new_rotation,rotation_matrix_l,rotation_matrix_l)

            #print calculate_error(new_rotation, t,l_reproject_pts,r_reproject_pts)
            ite -= 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(l_reproject_pts[:,0], l_reproject_pts[:,1], l_reproject_pts[:,2],c='r')
        ax.scatter(r_reproject_pts[:,0], r_reproject_pts[:,1], r_reproject_pts[:,2],c='b')
        plt.show()
        return (error,rotation_matrix_l.T,det_y_l , rotation_matrix_r.T, det_y_r)
    min_value = float("inf")
    min_result = None
    min_fov = None
    for d_fov_l in xrange(-1,1,1):
        for d_fov_r in xrange(-1,1,1):
            try:
                result = find_r_t(fov_l+d_fov_l/180.0 * math.pi, fov_r+d_fov_r/180.0 * math.pi)
            except:
                continue
            if result[0] < min_value:
                min_value = result[0]
                min_result = result
                min_fov = (fov_l+d_fov_l/180.0 * math.pi, fov_r+d_fov_r/180.0 * math.pi)



    return result,min_fov



def calculate_error(r,t,pts_a,pts_b):
    diff = 0.0
    for p_a, p_b in zip(pts_a,pts_b):
        diff += numpy.linalg.norm(r.dot(p_a)+t-p_b)
    return diff/len(p_a)


def find_rigid_transformation(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = numpy.mean(A, axis=0)
    centroid_B = numpy.mean(B, axis=0)
    
    # centre the points
    AA = A - numpy.tile(centroid_A, (N, 1))
    BB = B - numpy.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = numpy.transpose(AA) * BB

    U, S, Vt = numpy.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if numpy.linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return R, t


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


def vector_to_fisheye(px,py,pz, aperture, det_y = 0.0, rotation_matrix = numpy.eye(3)):
    """
    This function takes a 3D point on unit sephere, and map it to 2D fish eye
    coordinate.

    Args:
      px: x of 3D point
      py: y of 3D point
      pz: z of 3D point
      aperture: Fild of view of the image in radius.
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
      aperture: Fild of view of the image in radius.
      rotation_matrix: Not used for now

    Return:
      Matrix: A matrix same size as output_image.
    """
    print "input_image_size", input_image_size
    print "output_image_size", output_image_size
    if output_image_size[0] * 2 != output_image_size[1]:
        print "error output_image_size"
    else:
        image_map = numpy.zeros((output_image_size[0],output_image_size[1],2),dtype = numpy.float32)
        for x in xrange(output_image_size[1]):
            for y in xrange(output_image_size[0]):
                normal_x = lerp(x,0,output_image_size[1] - 1,-1,1)
                normal_y = lerp(y,0,output_image_size[0] - 1,-1,1)

                px,py,pz = equ_to_vector(normal_x, normal_y, r = 1000.0)
                normal_fish_x, normal_fish_y = vector_to_fisheye(px,py,pz,aperture, o_center_position,rotation_matrix)

                fish_x = lerp(normal_fish_x, -1,1, 0, input_image_size[1] -1)
                fish_y = lerp(normal_fish_y, -1,1, 0, input_image_size[0] -1)

                image_map[y,x] = fish_y,fish_x
        return image_map


def convert_fisheye_equ(input_image,output_image_name, output_image_size, aperture, rotation_matrix,o_center_position):
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
    cv2.imwrite(output_image_name,image)
    cv2.imshow("aaa",image)
    c = cv2.waitKey(0)
    return image

def test_rigid_motion():
    import random
    points_a = numpy.array([[1,0,0],[0,1,0],[0,0,1]],dtype=numpy.float32)
    points_b = numpy.array([[1,0,0],[0,1,0],[0,0,1]],dtype=numpy.float32)

    #print math.pi / 2.0 * random.random()
    rotation = rotation_matrix_x( math.pi*random.random())
    for i in xrange(len(points_b)):
        points_b[i] = rotation.dot(points_b[i])
    

    r,t = find_rigid_transformation(numpy.mat(points_b), numpy.mat(points_a))


    A2 = (r*points_b.T) + numpy.tile(t, (1, 3))
    A2 = A2.T

    # Find the error
    err = A2 - points_a
    print rotation-r


    print err
    err = numpy.multiply(err, err)
    err = numpy.sum(err)
    rmse = numpy.sqrt(err/3.0)
    print rmse

    


if __name__ == "__main__":
    input_image_r = cv2.imread("fisheye_r.jpg")
    input_image_l = cv2.imread("fisheye_l.jpg")
    #test_rigid_motion()
    
    
    #test_fisheye_to_equ()
    result, fov = dual_fisheye_calibrate(input_image_l,input_image_r,200 /180.0 *math.pi,200 /180.0 *math.pi)
    print fov[0]/ math.pi*180,fov[1]/math.pi*180
    error, r_l,t_l, r_r,t_r = result
    print error
    print r_l, t_l

    output_image = convert_fisheye_equ(input_image_l,"out_l.jpg", (2048,1024), 200.0 / 180.0 * PI, r_l, t_l)
    output_image = convert_fisheye_equ(input_image_r,"out_r.jpg", (2048,1024), 200.0 / 180.0 * PI, r_r, t_r)




