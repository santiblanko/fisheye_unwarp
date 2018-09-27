"""
This Module stores the routine to test routines in fisheye unwarp
"""
import math_util
from fisheye_unwarp import *

def rotation():
    for i in xrange(-8,8):
        r = math.pi / 4.0 * i
        r_m = rotation_matrix_z(r)
        print r_m
        input_image_l = cv2.imread("fisheye_l.jpg")
        output_image = convert_fisheye_equ(input_image_l,"out_l.jpg", (1024,512), 200.0/180*math.pi, r_m,numpy.array([0.0,0.0,0.0]))
        cv2.imwrite("rotation_test_{}.jpg".format(i*45), output_image)

def test_equ_to_fisheye():
    import random
    eps = 0.0000001
    rotation_matrix = math_util.rotation_matrix_x(60.0 / 180.0 * math.pi)
    r = 1000.0
    det_y = numpy.array([random.random(),random.random(),random.random()])
    fov = 200 /180.0 *math.pi
    for i in xrange(100):
        equ_coord = (random.random() - 0.5) * 2.0,(random.random() - 0.5) * 2.0
        px,py,pz = equ_to_vector(equ_coord[0],equ_coord[1],r)
        # print px,py,pz
        normal_fish_x, normal_fish_y = vector_to_fisheye(px,py,pz,fov, det_y,rotation_matrix.T)
        
        fish_coord = numpy.array([normal_fish_x, normal_fish_y])
        fi,theta = fisheye_coord_to_fi_theta(fish_coord,fov)
        p_coord = fi_theta_to_pxyz(fi,theta,det_y,r,rotation_matrix)
        # print p_coord
        x,y = pxyz_to_equ(p_coord)

        
        if abs( equ_coord[0] - x) > eps or abs(equ_coord[1] - y) > eps:
            print equ_coord
            print x,y
            print 'error'

def test_general():
    import random
    eps = 0.0000001
    rotation_matrix = math_util.rotation_matrix_z(math.pi).dot(math_util.rotation_matrix_x(random.random()*10*180 / math.pi))
    #rotation matrix for point set b, set a is eye
    eye = numpy.eye(3)
    r = 1000.0

    det_y_l = numpy.array([random.random(),random.random(),random.random()])
    det_y_r = -det_y_l
    fov = 200 /180.0 *math.pi
    fi_array = numpy.array([random.random()*math.pi for _ in xrange(10)])
    theta_array = numpy.array([math.pi + (random.random()-0.5)*10.0/180*math.pi\
                               for _ in xrange(10)])#180 +- 10 degree
    norm_equal_xy = map(lambda ro,t: (r*math.cos(ro),r*math.sin(t)), fi_array,theta_array)
    fish_coord_a = []
    fish_coord_b = []
    for x,y in norm_equal_xy:
        px,py,pz = equ_to_vector(x,y,r)
        fish_coord_a.append(numpy.array(vector_to_fisheye(px,py,pz,fov, det_y_l,eye)))
        fish_coord_b.append(numpy.array(vector_to_fisheye(px,py,pz,fov, det_y_r,rotation_matrix)))

    result,min_fov = dual_fisheye_calibrate(None,None, fov,fov, fish_coord_a,fish_coord_b)

    error, r_l,t_l, r_r,t_r = result

    print numpy.allclose(rotation_matrix.T,r_r)
    print numpy.allclose(t_l, det_y_l)

    for i, (x,y) in enumerate(norm_equal_xy):
        px,py,pz = equ_to_vector(x, y, r)
        fish_coord = vector_to_fisheye(px,py,pz,fov, t_r,r_r)
        if numpy.allclose(fish_coord, fish_coord_b[i]):
            print fish_coord, fish_coord_b[i]


    #1. randomly choose three points with correct theta, phi
    #2. generate second rotation matrix and translation(first one is eye)
    #3. generate each points fisheye point
    #4. throw it in dual fish eye calibrate
    #5. check if mat is the same

def test_rigid_motion():
    import random
    points_a = numpy.array([[1,0,0],[0,1,0],[0,0,1]],dtype=numpy.float32)
    points_b = numpy.array([[1,0,0],[0,1,0],[0,0,1]],dtype=numpy.float32)

    #print math.pi / 2.0 * random.random()
    rotation = math_util.rotation_matrix_x( math.pi*random.random())
    for i in xrange(len(points_b)):
        points_b[i] = rotation.dot(points_b[i])
    

    r,t = math_util.find_rigid_transformation(numpy.mat(points_b), numpy.mat(points_a))


    A2 = (r*points_b.T) + numpy.tile(t, (1, 3))
    A2 = A2.T

    # Find the error
    err = A2 - points_a


    print err
    err = numpy.multiply(err, err)
    err = numpy.sum(err)
    rmse = numpy.sqrt(err/3.0)
    print rmse

test_equ_to_fisheye()
test_rigid_motion()
test_general()