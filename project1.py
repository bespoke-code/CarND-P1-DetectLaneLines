import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import os
import math


# Functions for image manipulation
# --------------------------------
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    plt.imshow(img)
    plt.title("Draw_lines started")
    plt.show()
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    plt.imshow(img)
    plt.title("Draw_lines finished")
    plt.show()

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# The main function - this is where the fun happens!
# --------------------------------------------------
def process_image(originalImg):
    imshape = originalImg.shape
    ysize = imshape[0]
    xsize = imshape[1]

    # Show the image (test)
    plt.imshow(originalImg)
    plt.show()

    # Convert to grayscale and apply blur (denoise)
    kernelSize = 5
    grayImg = cv2.cvtColor(originalImg, cv2.COLOR_RGB2GRAY)
    processedImg = cv2.GaussianBlur(grayImg, (kernelSize, kernelSize), 0)
    plt.imshow(processedImg)
    plt.title("After desaturation and Gaussian blur. Next: Canny")
    plt.show()

    # Apply Canny filter
    processedImg = canny(processedImg, 50, 150)

    # Apply mask in a region
    vertices = np.array([[(2 * xsize / 100, ysize),  # bottom left
                          ((15 * xsize) / 32, (17 * ysize) / 32),  # top left
                          ((17 * xsize) / 32, (17 * ysize) / 32),  # top right
                          (98 * xsize / 100, ysize)  # bottom right
                          ]],
                        dtype=np.int32
                        )
    processedImg = region_of_interest(processedImg, vertices)

    # Apply Hough Transform (detect lines)
    lines = hough_lines(processedImg, rho=1, theta=np.pi / 180, threshold=6, min_line_len=30, max_line_gap=16)
    plt.imshow(processedImg)
    plt.title("Processed image after Canny & Hough Transform")
    plt.show()

    # Get a blank image copy to overlay the lines on
    line_image = np.copy(processedImg) * 0
    plt.imshow(line_image)
    plt.title("Empty image copy for furter action. Next: Draw Lines")
    plt.show()
    # Here comes the part where we must extrapolate both left and right lines
    # to connect from the closest to the furthest point in the region of interest

    # TODO: Add algorithm here

    #draw_lines(line_image, lines)

    line_image_colour = np.dstack((line_image, line_image, line_image))
    # Combine original and lines image
    alpha = 0.8
    beta = 1.0
    l = 0.0
    finalImg = weighted_img(originalImg, line_image_colour, alpha, beta, l)
    plt.imshow(finalImg)
    plt.title("Final Image after analysis")
    plt.show()
    return finalImg

# Processing a single image
# -------------------------
def main_on_images(imgPath):
    # Import image
    originalImg = mpimg.imread(imgPath)

    # Do magic
    finalImg = process_image(originalImg)

    # Save (?) and review
    # mpimg.imsave(finalImg)
    plt.imshow(finalImg)
    plt.show()


# Even more fun here!
# Processing a video file
# -----------------------
def main_on_video(inputVideoPath, outputVideoPath):
    # Import a video file to work on
    originalVid = VideoFileClip(inputVideoPath)

    # Apply the single image processing pipeline to each frame of the video
    outputVid = originalVid.fl_image(process_image)

    outputVid.write_videofile(outputVideoPath, audio=False)
    print("Video processed and saved successfully at {location}.".format(location=outputVideoPath))

# Why not call this from the command line?
# ----------------------------------------
if __name__ == "__main__":
    #imgDir = "/home/andrej/git/CarND-P1-DetectLaneLines/test_images"
    #images = os.listdir(imgDir)
    #for image in images:
    #    main_on_images(imgDir + '/' + image)
    main_on_images("/home/andrej/git/CarND-P1-DetectLaneLines/test_images/solidWhiteCurve.jpg")
    print("Done!")