import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from src.data.analysis import Circle, median_filter, subtract_background

def img_for_circle_detection(input: npt.NDArray, threshold: float = 0.3) -> npt.NDArray:
    logging.info('Before setting threshold ' +
                 f'min {np.nanmin(input)}, mean {np.nanmean(input):3.3f}, max {np.nanmax(input)}')
    output = np.zeros(shape=input.shape, dtype='uint8')
    threshold_level = np.nanpercentile(a=input, q=30)  # percentile at 30%
    output[input < threshold_level] = 255
    output[np.isnan(input)] = 255
    logging.info('After setting threshold ' +
                 f'min {np.min(output)}, mean {np.mean(output):3.3f}, max {np.max(output)}')
    return output


def find_circle_hough_method(input: npt.NDArray) -> Circle:
    logging.info(f'Detector circle not provided, calculating with Hough method')
    hough_results = cv2.HoughCircles(input, cv2.HOUGH_GRADIENT, dp=1, minDist=10000, param1=10, param2=0.9, minRadius=300, maxRadius=600)
    logging.info(f'Hough method results {hough_results}')
    result_circle = Circle()
    if hough_results is None:
        print("No detector found by Hough method")
    elif hough_results.shape[0] > 1:
        print("More than one shape found by Hough method")
    elif hough_results.shape[0] == 1 and hough_results.shape[1] == 1:
        # explicit conversion to float is needed to ensure proper JSON serialisation
        # hough_results is a numpy float32 array and float32 is not JSON serialisable
        result_circle = Circle(
            x=float(hough_results[0, 0, 0]),
            y=float(hough_results[0, 0, 1]),
            r=float(hough_results[0, 0, 2]),
        )
        logging.info(f'Detected circle {result_circle}')
    return result_circle


def find_detector(input: npt.NDArray,
                  img_bg: Optional[npt.NDArray] = None,
                  threshold: float = 0.3,
                  const_bg: float = 0,
                  circle: Optional[Circle] = None,
                  gpu: bool = False) -> Tuple[Circle, npt.NDArray]:
    '''
    find the detector on the live-view image,
    returns position of detector center and its radius
    and image with detector
    if circle (center of detector position and radius) is not provided it will be calculated and returned
    detector radius and center position are rounded to integer numbers (TODO why???)
    '''

    # make a copy of the image, in order to exlude modification of the original data
    logging.info(
        'Original image ' +
        f'shape {input.shape}, min {np.nanmin(input)}, mean max {np.nanmean(input):3.3f}, max {np.nanmax(input)}')

    # MF (median filter) on original image
    img_mf = median_filter(input=input, gpu=gpu)

    # MF (median filter) on background image
    img_bg_mf = None
    if img_bg is not None:
        img_bg_mf = median_filter(input=img_bg, gpu=gpu)

    # Background remove (constant BG (CBG) and imgBG) applied after median filters
    img_bg_sub = subtract_background(input=img_mf, img_bg=img_bg_mf, const_bg=const_bg, gpu=gpu)

    # TH (threshold)
    img_thres = img_for_circle_detection(input=img_bg_sub, threshold=threshold)

    # find detector
    result_circle = find_circle_hough_method(img_thres)

    return result_circle, img_thres