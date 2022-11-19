import dataclasses
import datetime
import json
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import ndimage


@dataclass(frozen=True)
class Circle:
    '''Read-only data class for storing center position and radius of a circle'''
    x: float = float('nan')
    y: float = float('nan')
    r: float = float('nan')

    @property
    def proper(self):
        return self.x > -np.inf and self.y > -np.inf and self.r >= 0.

    def save_json(self, path):
        with open(path, 'w') as json_file:
            json.dump(dataclasses.asdict(self), json_file)

    @classmethod
    def from_json(cls, path : str):
        circle = cls()
        with open(path, 'r') as openfile:
            json_data = json.load(openfile)
            circle = cls(**json_data)
        return circle

    def section_x(self, x : float) -> Tuple[float, float]:
        under_sqrt = self.r**2 - (x - self.x)**2
        result = (float('nan'), float('nan'))        
        if under_sqrt >= 0:
            result = (self.y - math.sqrt(under_sqrt), self.y + math.sqrt(under_sqrt))
        return result

    def section_y(self, y : float) -> Tuple[float, float]:
        under_sqrt = self.r**2 - (y - self.y)**2
        result = (float('nan'), float('nan'))        
        if under_sqrt >= 0:
            result = (self.x - math.sqrt(under_sqrt), self.x + math.sqrt(under_sqrt))
        return result

def read_tiff_img(file_path: str, border_px: int = 50) -> npt.NDArray:
    '''
    read tif image and add a border filled with NaN
    new image will have bigger size (by 2 x border_px in each direction) than original tiff
    Note: input file contains 16-bit integers, but we intentionally cast the output to the 32-bit float array
          the reason is that we want to pad the array with NaNs which are transpartent to methods like min/max/mean
          there is no NaN for integers, therefore we are forced to use floats despite int-like values
    '''
    logging.info(f'Reading file {file_path}')
    raw_img = plt.imread(file_path).astype('float32')
    logging.info(f'Original image shape: {raw_img.shape}, min value {raw_img.min()}, max value {raw_img.max()}')
    padded_img = np.pad(raw_img, pad_width=border_px, constant_values=np.nan)
    logging.info(f'Padded image shape: {padded_img.shape}')
    return padded_img


def create_circular_mask(img: npt.NDArray, circle_px: Circle) -> npt.NDArray:
    '''create a circular mask of the same resolution as the image.'''

    y_grid, x_grid = np.ogrid[:img.shape[0], :img.shape[1]]
    # x_grid has shape (img.shape[0], 1)
    # y_grid has shape (1, img.shape[1])

    dist_from_center_squared = (x_grid - circle_px.x)**2 + (y_grid - circle_px.y)**2
    # broadcasting will guarantee that the formula above will give us shape (img.shape[0], img.shape[1])

    circ_mask = dist_from_center_squared <= circle_px.r**2
    return circ_mask


def default_circular_mask(img: npt.NDArray) -> npt.NDArray:
    '''
    create a circular mask with circle in the middle of the image
    and maximum possible radius to be fully enclosed in the image
    '''
    center_x = img.shape[0] / 2
    center_y = img.shape[1] / 2
    radius = min(center_x, center_y)

    return create_circular_mask(img, Circle(x=center_x, y=center_y, r=radius))


def median_filter(input: npt.NDArray, size: int = 10, gpu: bool = False) -> npt.NDArray:
    '''apply median filter'''
    logging.info('Before median filter ' +
                 f'min {np.nanmin(input)}, mean max {np.nanmean(input):3.3f}, max {np.nanmax(input)}')
    output = np.empty(shape=1)
    if gpu:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import median_filter as median_filter_gpu
            output = median_filter_gpu(cp.asarray(input), size=size).get()
        except (ModuleNotFoundError, ImportError):
            logging.warning('GPU mode selected and no `cupy` library installed')
            return np.full_like(input, np.nan, dtype=np.double)
    else:
        output = ndimage.median_filter(input, size=size)
    logging.info('After median filter ' +
                 f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}')
    return output


def subtract_background(input: npt.NDArray,
                        img_bg: Optional[npt.NDArray] = None,
                        const_bg: float = 0,
                        gpu: bool = False) -> npt.NDArray:
    '''Background remove (constant BG (CBG) and imgBG)
    # assume zero background if no `img_bg` option provided'''

    output = input.copy()

    if img_bg is not None:
        output -= img_bg
    logging.info('After background image subtraction ' +
                 f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}')

    # subtract constant background factor
    output -= const_bg
    logging.info('After constant background factor subtraction ' +
                 f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}')

    # set all pixels with negative values to zero
    # optionally use: `np.clip(a=img_det, a_min=0., out=img_det)`
    output[output < 0] = 0
    logging.info('After removing pixels with negative value ' +
                 f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}')

    return output


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


def get_line_circle(img: npt.NDArray, circ: Circle, step_angle_deg: float = 1.0, max_angle_deg: float = 360.0):
    # get a profile along the circle defined as [x,y,r]

    img_line = img.copy()
    img_line[np.isnan(img_line)] = 0
    logging.info(f'Input image shape {img_line.shape} min {img_line.min():3.3f}, max {img_line.max():3.3f}')

    array_of_angles_deg = np.arange(start=0, stop=max_angle_deg, step=step_angle_deg)
    logging.info(f'Array of angles, step {step_angle_deg}, max angle {max_angle_deg}')
    points_on_circle_px = np.array([(np.sin(np.deg2rad(-angle_deg)) * circ.r + circ.x,
                                     -np.cos(np.deg2rad(-angle_deg)) * circ.r + circ.y)
                                    for angle_deg in array_of_angles_deg])
    logging.info(f'Points on circle # {points_on_circle_px.size}, shape {points_on_circle_px.shape}')
    logging.info(f'Points on circle X from {points_on_circle_px[:,0].min()} to {points_on_circle_px[:,0].max()}')
    logging.info(f'Points on circle Y from {points_on_circle_px[:,1].min()} to {points_on_circle_px[:,1].max()}')
    # alternative:
    #
    '''
    circPoints_px = np.array(
        [
            (np.sin(np.deg2rad(-angle_deg)), -np.cos(np.deg2rad(-angle_deg)))
                              for angle_deg in array_of_angles_deg
                              ]
                              )
    circPoints_px *= circ.r
    circPoints_px[0,:] += circ.x
    circPoints_px[1,:] += circ.y
    '''

    values_on_circle = ndimage.map_coordinates(img_line.T, points_on_circle_px.T)
    logging.info(f'Values on circle min {values_on_circle.min():3.3f}, max {values_on_circle.max():3.3f}')
    return array_of_angles_deg, values_on_circle


def get_angle_with_min_value(array_of_angles_deg: npt.NDArray,
                             values_on_circle: npt.NDArray,
                             median_filter_size: int = 10) -> Tuple[float, npt.NDArray]:
    min_value = ndimage.median_filter(input=values_on_circle, size=median_filter_size)
    min_value_angle_deg = array_of_angles_deg[np.nanargmin(min_value)]
    return min_value_angle_deg, min_value

def get_mean_std(data: npt.NDArray, circle : Circle) -> Tuple[float, float]:
    mask = create_circular_mask(img=data, circle_px=circle)
    mean = np.nanmean(data[mask]).astype(float)
    std = np.nanstd(data[mask]).astype(float)
    return mean, std

def get_timestamp(filepath : str) -> datetime.datetime:
    metada_contents = ''
    with open(filepath, 'r', encoding="ISO-8859-1") as metadata_file:
        metada_contents = metadata_file.read()

    parsed_json = json.loads(metada_contents)
    time_str = parsed_json['Summary']['Time']  # or 'StartTime'
    result = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S %z')
    return result
