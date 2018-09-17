import numpy
import math

def rotation_matrix_decompose(r):
    return numpy.array( (math.atan2(r[2][1],r[2][2]),\
                        math.atan2(-r[2][0],math.sqrt(r[2][1]*r[2][1]+r[2][2]*r[2][2])),\
                        math.atan2(r[1][0],r[0][0])))


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
    # if not(x0 <= x <= x1):
    #     print x

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
def rotation_matrix_z(theta):
    return numpy.array([
        [math.cos(theta), -math.sin(theta),0],
        [math.sin(theta), math.cos(theta),0],
        [0,0,1 ]
    ])

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