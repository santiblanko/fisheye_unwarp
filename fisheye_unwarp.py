import cv2
import numpy
import math

PI = math.pi

def lerp(x, x0,x1,y0,y1):
	x, x0,x1,y0,y1 = map(float, (x, x0,x1,y0,y1))
	return y0+(x-x0)*(y1-y0)/(x1-x0)


def equ_to_vector(x,y):
	lon = x * PI
	lat = y * PI / 2.0

	Px = math.cos(lat) * math.cos(lon)
	Py = math.cos(lat) * math.sin(lon)
	Pz = math.sin(lat)
	return Px,Py,Pz

def vector_to_fisheye(px,py,pz, aperture):
	r = 2 * math.atan2(math.sqrt(px*px + pz*pz),py)/aperture
	theta = math.atan2(pz,px)

	x = r * math.cos(theta)
	y = r * math.sin(theta)
	return x,y

def generate_map(input_image_size, output_image_size, aperture, rotation_matrix):
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

				px,py,pz = equ_to_vector(normal_x, normal_y)
				normal_fish_x, normal_fish_y = vector_to_fisheye(px,py,pz,aperture)

				fish_x = lerp(normal_fish_x, -1,1, 0, input_image_size[1] -1)
				fish_y = lerp(normal_fish_y, -1,1, 0, input_image_size[0] -1)

				image_map[y,x] = fish_y,fish_x
		return image_map


def convert_fisheye_equ(input_image, output_image_size, aperture, rotation_matrix):
    image = numpy.zeros((output_image_size[1],output_image_size[0],3),dtype = numpy.uint8)
    image_map = generate_map(input_image.shape, image.shape, aperture, rotation_matrix)

    cv2.remap(src=input_image, dst=image, map1=image_map, map2=None,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("image.jpg",image)
    cv2.imshow("aaa",image)
    c = cv2.waitKey(0)
    return image

if __name__ == "__main__":
    input_image = cv2.imread("fisheye_l.jpg")
    output_image = convert_fisheye_equ(input_image, (4096,2048), 200.0 / 180.0 * PI, None)




