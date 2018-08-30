"""
This module converts fisheye image into panoramic video. 
"""

import cv2
import numpy
import numpy.linalg
import math

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

def vector_to_r_theta_with_delta_y(px,py,pz, dy, aperture):
    vec = numpy.array([px,py-dy,pz])
    px,py,pz = vec/numpy.linalg.norm(vec)
    r = 2 * math.atan2(math.sqrt(px*px + pz*pz),py)/aperture
    theta = math.atan2(pz,px)
    return r, theta


def vector_to_fisheye(px,py,pz, aperture, det_y = 0.0):
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
    r,theta = vector_to_r_theta_with_delta_y(px,py,pz, det_y,aperture)

    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x,y


def generate_map(input_image_size, output_image_size, aperture, rotation_matrix):
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

                px,py,pz = equ_to_vector(normal_x, normal_y, 100.0)
                normal_fish_x, normal_fish_y = vector_to_fisheye(px,py,pz,aperture, -13.86/2.0)

                fish_x = lerp(normal_fish_x, -1,1, 0, input_image_size[1] -1)
                fish_y = lerp(normal_fish_y, -1,1, 0, input_image_size[0] -1)

                image_map[y,x] = fish_y,fish_x
        return image_map


def convert_fisheye_equ(input_image, output_image_size, aperture, rotation_matrix):
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
    image_map = generate_map(input_image.shape, image.shape, aperture, rotation_matrix)

    cv2.remap(src=input_image, dst=image, map1=image_map, map2=None,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("image.jpg",image)
    cv2.imshow("aaa",image)
    c = cv2.waitKey(0)
    return image

if __name__ == "__main__":
    input_image = cv2.imread("fisheye_left.jpg")
    output_image = convert_fisheye_equ(input_image, (2048,1024), 200.0 / 180.0 * PI, None)




