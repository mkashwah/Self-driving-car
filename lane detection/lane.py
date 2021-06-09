import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int (y1*(3/5))
    x1 = int((y1- intercept)/slope)
    x2 = int((y2- intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])


def canny(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #transfer to gray because it makes it easier to have 1 channel grey rather than 3 rgb
    blur1 = cv2.GaussianBlur(grey_img, (5,5), 0)
    canny = cv2.Canny(blur1, 50, 150)    #set the sensitivity of edge detection
    return canny

def display_lines(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_img, (x1,y1), (x2,y2), (255,0,0), 10)

    return line_img
#identifying region of interest:
def region_of_interest(img):
    height = img.shape[0]
    polys = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polys, 255)
    masked_img = cv2.bitwise_and(img, mask)    #implements the bitwise multiplicartion
    return masked_img

image1 = cv2.imread('test_image.jpg')
lane_img = np.copy(image1)   #copy the image array into a new variable
canny_img = canny(lane_img)
cropped_img = region_of_interest(canny_img)
# mask = np.zeros_like(canny)
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)


averaged_lines = average_slope_intercept(lane_img, lines)
line_img = display_lines(lane_img, averaged_lines)
combo1_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
cv2.imshow('final image', combo1_img)
cv2.imshow('cropped image', line_img)

cv2.waitKey(0)




# cv2.imshow('result', image1)   #original image
# cv2.imshow('grey result', grey_img)
# cv2.imshow('blur1', blur1)   #blurred image
# cv2.imshow('both', blur1, grey_img)   #doesn't work takes only 2 arguments
# plt.imshow(canny)    #shows in a plot
# cv2.imshow('blur3', blur3)
# cv2.imshow('blur4', blur4)
# cv2.imshow('blur5', blur5)
#plt.show()   #keeps plot on screen
