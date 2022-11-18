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

    def section_x(self, x : float) -> Tuple[float]:
        under_sqrt = self.r**2 - (x - self.x)**2
        result = (float('nan'), float('nan'))        
        if under_sqrt >= 0:
            result = (self.y - math.sqrt(under_sqrt), self.y + math.sqrt(under_sqrt))
        return result

    def section_y(self, y : float) -> Tuple[float]:
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
                  gpu: bool = False) -> Circle:
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
                             median_filter_size: int = 10) -> Tuple[float]:
    min_value = ndimage.median_filter(input=values_on_circle, size=median_filter_size)
    min_value_angle_deg = array_of_angles_deg[np.nanargmin(min_value)]
    return min_value_angle_deg, min_value


def cropToMask(img, circMask):
    # crop image to mask
    imgMask = create_circular_mask(img, circMask)
    imgMask = imgMask.astype('uint8') * 255
    maskRect = cv2.boundingRect(imgMask)
    return (img[maskRect[1]:(maskRect[1] + maskRect[3]), maskRect[0]:(maskRect[0] + maskRect[2])], maskRect)


def correctImg(img, imgFF=None, imgBG=None, CBG=0, medianFilter=1, circMask=None):
    imgProc = img.copy()

    # create a mask
    if circMask is not None:
        imgMask = create_circular_mask(img, circMask)
    else:
        imgMask = np.ones(img.shape).astype('bool')

    # BG remove
    if imgBG is not None:
        imgProc = imgProc - imgBG
    imgProc -= CBG
    imgProc[imgProc < 0] = 0

    # FF correction (https://en.wikipedia.org/wiki/Flat-field_correction)
    if imgFF is not None:
        imgFFcorr = imgFF - imgBG
        imgFFcorr = imgFFcorr / np.nanmean(imgFFcorr[imgMask])
        imgProc[imgMask] = (imgProc[imgMask] / imgFFcorr[imgMask])

    # MF median filter
    if medianFilter is not None and medianFilter != 1:
        imgProc = ndimage.median_filter(imgProc, size=medianFilter)

    # apply mask
    imgProc[imgProc < 0] = 0
    imgProc[~imgMask] = np.nan

    return imgProc


def rotateImg(img, angle):
    # rotate image around its centre
    img[np.isnan(img)] = -1
    if angle != 0:
        img = ndimage.rotate(img, angle, reshape=False, cval=np.nan, prefilter=False)
    img[img < 0] = np.nan
    return img


def resizeImg(img, size):
    # resample image
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LANCZOS4)
    img[img < 0] = 0
    return img


def fillHoles(img):

    imgInPaintMask = img.copy()
    imgInPaintMask[imgInPaintMask > 0] = 1
    imgInPaintMask[imgInPaintMask == 0] = 0
    imgInPaintMask[np.isnan(imgInPaintMask)] = 1
    imgInPaintMask = imgInPaintMask.astype('bool')
    imgInPaintMask = np.invert(imgInPaintMask)
    imgInPaintMask = imgInPaintMask.astype('uint8')
    img = cv2.inpaint(img, imgInPaintMask, 3, cv2.INPAINT_TELEA)
    return (img)


def correctIRI(imgMeas, circMeas, imgIRI, circIRI, rotateIRI=0):
    imgMeasCrop, rectMeasCrop = cropToMask(imgMeas, circMeas)
    imgIRI, rectIRI = cropToMask(imgIRI, circIRI)
    imgIRI = resizeImg(imgIRI, size=(rectMeasCrop[2], rectMeasCrop[3]))
    imgIRI = fillHoles(imgIRI)
    imgIRI = rotateImg(imgIRI, angle=0)
    IRFmean = np.nanmean(imgIRI)
    IRFstd = np.nanstd(imgIRI)
    imgIRI = imgIRI / IRFmean
    imgMeasIRI = imgMeasCrop / imgIRI

    return (imgMeasIRI, IRFmean, IRFstd)


def label_text(data: npt.NDArray, title : str) -> str:
    result = '\n'.join((
    title,
    f'mean = {np.nanmean(data):.3f}',
    f'median = {np.nanmedian(data):.3f}',
    '\n',
    f'stddev = {np.nanstd(data):.3f}',
    f'stddev / mean = {100. * np.nanstd(data) / np.nanmean(data):.3f} %',
    '\n',
    f'min = {np.nanmin(data):.3f}',
    f'max = {np.nanmax(data):.3f}',
    ))
    return result
    

def min_max_area_loc(data: npt.NDArray, circle_px: Circle, window_size : int = 10) -> Tuple[Tuple[float]]:
    N,M = window_size, window_size
    P,Q = data.shape
    mask_for_circle = create_circular_mask(img=data, circle_px=circle_px)

    meds = ndimage.median_filter(data, size=(M,N))
    meds[~mask_for_circle] = np.nan
    data_medfilt = meds[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]

    max_idx = np.unravel_index(np.nanargmax(data_medfilt), data_medfilt.shape)
    max_center = max_idx[0]+window_size,max_idx[1]+window_size

    min_idx = np.unravel_index(np.nanargmin(data_medfilt), data_medfilt.shape)
    min_center = min_idx[0]+window_size,min_idx[1]+window_size

    return (min_center, max_center)
    

def plot_data(data : npt.NDArray, path : str, circle_px: Circle = None, details : bool = False):
    circle = circle_px
    if not circle:
        circle = Circle(x=data.shape[0]/2,y=data.shape[0]/2,r=250)

    if details:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16,10), constrained_layout=True)
        ax = axes[0][0]
    else:
        fig, ax = plt.subplots(figsize=(16,10), constrained_layout=True)
    
    # don't plot top 5% to avoid hot pixels
    data_for_plotting = np.clip(data, a_min=None, a_max=np.nanpercentile(a=data, q=95))
    pos0 = ax.imshow(data_for_plotting, cmap='terrain', interpolation='None');
    plt.colorbar(pos0, ax=ax, shrink=0.4);

    mask_for_circle = create_circular_mask(img=data, circle_px=circle)
    title_for_circle = f'circle at {circle.x},{circle.y}\n     radius {circle.r:.1f} : \n'
    text_for_circle = label_text(data=data[mask_for_circle], title=title_for_circle)

    title_for_image = f'full image : \n\n'
    text_for_image = label_text(data=data, title=title_for_image)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    if not details:
        # place a text box for circular area
        ax.text(0.15, 0.95, text_for_circle, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', 
            horizontalalignment='left',
            bbox=props)

        # place a text box for image area
        ax.text(0.65, 0.95, text_for_image, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', 
            horizontalalignment='left',
            bbox=props)

    ax.add_artist(plt.Circle(xy=(circle.x, circle.y), 
        radius=circle.r, 
        color='black', 
        fill=False, 
        transform=ax.transData))

    mean_in_circle = np.nanmean(data[mask_for_circle])
    std_in_circle = np.nanstd(data[mask_for_circle])
    bins=np.arange(start=int(mean_in_circle - 5*std_in_circle), stop=int(mean_in_circle + 5*std_in_circle),step=1)
    if bins.size < 30:
        bins=np.linspace(start=mean_in_circle - 5*std_in_circle, stop=mean_in_circle + 5*std_in_circle,num=100)

    if details:
        axes[1][1].hist(data[mask_for_circle].flatten(), bins=bins, histtype='step', label='original')
        axes[1][1].grid()

        section_cut_x = circle.x + circle.r/2
        section_cut_y = circle.y + circle.r/2
        window_size = 10
        min_center, max_center = min_max_area_loc(data, circle_px = circle, window_size=window_size)
        min_center, max_center
        axes[0][0].set_xlabel("X")
        axes[0][0].set_ylabel("Y")

        axes[0][0].axvline(section_cut_x, color='blue')
        axes[0][0].axvline(min_center[0], color='green')
        axes[0][0].axvline(max_center[0], color='red')

        axes[0][0].axhline(section_cut_y, color='blue')
        axes[0][0].axhline(min_center[1], color='green')
        axes[0][0].axhline(max_center[1], color='red')

        def section(data, location, window_size, axis : int, circle_px : Circle = None):
            start = int(location - window_size/2)
            stop = int(location + window_size/2)    
            if axis == 1: # along Y, fixed X
                result = np.average(data[:,start:stop], axis=1)
                if circle_px:
                    lower_y, upper_y = circle.section_x(x=0.5*(start+stop))
                    result[0:int(lower_y)] = np.nan
                    result[int(upper_y):] = np.nan
            else:  # along X, fixed Y
                result = np.average(data[start:stop,:], axis=0)
                if circle_px:
                    lower_x, upper_x = circle.section_y(y=0.5*(start+stop))
                    result[0:int(lower_x)] = np.nan
                    result[int(upper_x):] = np.nan
            return result

        def abs2perc(x):
            # y = 100 - 100 * x / mean_in_circle
            return 100. - 100. * x / mean_in_circle

        def perc2abs(x):
            # x = 100 - 100 * y / mean_in_circle
            # 100 * y / mean_in_circle = 100 - x
            # 100 * y = mean_in_circle * (100 - x)
            # y = mean_in_circle * (100 - x) / 100
            return mean_in_circle * (100. - x) / 100.

        axes[0][1].plot(section(data, section_cut_x, window_size, 1, circle), label='ref', color='blue')
        axes[0][1].plot(section(data, min_center[0], window_size, 1, circle), label='min', color='green')
        axes[0][1].plot(section(data, max_center[0], window_size, 1, circle), label='max', color='red')
        axes[0][1].axvline(section_cut_y, color='blue')
        axes[0][1].axvline(min_center[1], color='green')
        axes[0][1].axvline(max_center[1], color='red')
        axes[0][1].set_xlabel("Y")
        secax = axes[0][1].secondary_yaxis('right', functions=(abs2perc, perc2abs))
        secax.set_ylabel(' (value-mean)/mean [%] ')
        axes[0][1].grid()
        axes[0][1].legend()

        axes[1][0].plot(section(data, section_cut_y, window_size, 0, circle), label='ref', color='blue')
        axes[1][0].plot(section(data, min_center[1], window_size, 0, circle), label='min', color='green')
        axes[1][0].plot(section(data, max_center[1], window_size, 0, circle), label='max', color='red')
        axes[1][0].axvline(section_cut_x, color='blue')
        axes[1][0].axvline(min_center[0], color='green')
        axes[1][0].axvline(max_center[0], color='red')
        axes[1][0].set_xlabel("X")
        secax = axes[1][0].secondary_yaxis('right', functions=(abs2perc, perc2abs))
        secax.set_ylabel(' (value-mean)/mean [%] ')
        axes[1][0].grid()
        axes[1][0].legend()

    if path:
        fig.savefig(path)

    if details:
        return fig, axes
    return fig, ax

def get_mean_std(data: npt.NDArray, circle : Circle) -> Tuple[float, float]:
    mask = create_circular_mask(img=data, circle_px=circle)
    mean = np.nanmean(data[mask])
    std = np.nanstd(data[mask])
    return mean, std

def get_timestamp(filepath : str) -> datetime:
    metada_contents = ''
    with open(filepath, 'r', encoding="ISO-8859-1") as metadata_file:
        metada_contents = metadata_file.read()

    parsed_json = json.loads(metada_contents)
    time_str = parsed_json['Summary']['Time']  # or 'StartTime'
    result = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S %z')
    return result
