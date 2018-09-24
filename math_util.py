"""
This module provides necessary math utilties for fisheye unwarp. 
"""
import numpy
import math

def rotation_matrix_decompose(r):
    """
    This function helps analyse rotation matrix, by decompose it into rotation on x,y,z axis. Please note the eular
    angles provided is rotated in z, y, x order. See: http://nghiaho.com/?page_id=846.

    Args:
      r: Rotation matrix.

    Returns:
      x: Rotation on x axis, in raids
      y: Rotation on y axis, in raids
      z: Rotation on z axis, in raids
    """
    return numpy.array( (math.atan2(r[2][1],r[2][2]),\
                        math.atan2(-r[2][0],math.sqrt(r[2][1]*r[2][1]+r[2][2]*r[2][2])),\
                        math.atan2(r[1][0],r[0][0])))


def lerp(x, x0,x1,y0,y1):
    """
    This function is a helper function to normalize values.
    Mathematically, this function does a linear interpolation for x from
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
    # if not(x0 <= x <= x1):
    #     print x

    x, x0,x1,y0,y1 = map(float, (x, x0,x1,y0,y1))
    return y0+(x-x0)*(y1-y0)/(x1-x0)

def solve_quadratic(a,b,c):
    """
    This function solves quadratic, ax^2 + bx + c = 0. Please note it only gives real solution(s).

    Args:
        a: Coefficient of the quadratic.
        b: Coefficient of the quadratic.
        c: Coefficient of the quadratic.

    Returns:
        x1,x2: Two solutions, x1 will always be larger. if there's no solution it will return None
    """
    if b*b - 4 * a * c < 0:
        return None# no real solution
    else:
        x1 = (-b + math.sqrt(b*b-4 * a * c))/ (2 * a)
        x2 = (-b - math.sqrt(b*b-4 * a * c))/ (2 * a)
        return x1,x2

def rotation_matrix_x(theta):
    """
    This function generates rotation matrix that rotates on x axis.

    Args:
        theta: Amount of rotation in radius.

    Returns:
        r: Rotation matrix
    """
    return numpy.array([
        [1,0,0],
        [0,math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ])
def rotation_matrix_z(theta):
    """
    This function generates rotation matrix that rotates on z axis.

    Args:
        theta: Amount of rotation in radius.

    Returns:
        r: Rotation matrix
    """
    return numpy.array([
        [math.cos(theta), -math.sin(theta),0],
        [math.sin(theta), math.cos(theta),0],
        [0,0,1 ]
    ])

def find_rigid_transformation(A, B):
    """
    This function finds optimal rigid transformation on point set A and point set B. Please note you should use matrix
    rather than array. Result should satisfy R*A + t = B.

    Args:
        A: A Nxm numpy **matrix**. N is the count of sample points, m is the dimension of each sample.
        B: A Nxm numpy **matrix**.

    Returns:
       R: Rotation matrix.
       t: Translation of the points.
    """
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