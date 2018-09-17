import numpy as np
import cv2 as cv

i = 0

def find_checkerboard(mask_rect, image, pattern_size = (6, 3)):
    """
    This function finds points on checkerbord. 

    Args:
      mask_rect: (x,y,w,h), Part of the image that is used to find checkerboard,\
      x,y is the top left start point. w,h is the width and height of the image.
      image: Gray scale image
      pattern_size: The size of checker board.

    Return:
      corners: The founded checker board. \
      Note, if mask is set than the corners is compensated for masking.
    """
    global i
    img_h,img_w = image.shape[:2]

    subImage = image[mask_rect[1]:min(mask_rect[1]+mask_rect[3],img_h),\
                     mask_rect[0]:min(mask_rect[0]+mask_rect[2],img_w)]
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    #find corners
    found, corners = cv.findChessboardCorners(subImage, pattern_size)

    #if found refine by subpix
    #cv.imshow("aaa", subImage)
    #cv.waitKey(3000)
    if found:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(subImage, corners, (5, 5), (-1, -1), term)
        #vis = cv.cvtColor(subImage, cv.COLOR_GRAY2BGR)
        
        #cv.drawChessboardCorners(vis, pattern_size, corners, found)
        #cv.imshow("aaa", vis)
        #cv.imwrite('checkerboard_{0}.png'.format(i), vis)
        i+= 1
        
        corners[:,:,0:2] += mask_rect[:2]
        return corners
    else:
        return None

if __name__ == "__main__":

    image = cv.imread("fisheye_l.jpg")
    h,w = image.shape[:2]
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    find_checkerboard(np.array([int(w/2.0),int(h/2.0),int(w/2.0),int(h/2.0)]),gray_img )
