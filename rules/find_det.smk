import json
import dataclasses
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from src.visualisation.plot import plot_data
from src.data.analysis import (
    read_tiff_img,
    subtract_background,
    Circle,
    create_circular_mask,
)
from src.data.detector import (
    img_for_circle_detection,
    find_circle_hough_method
)
from src.data.alignment import (
    get_line_circle,
    get_angle_with_min_value,
)

rule read_tiff:
    input:
        data="data/raw/foils/{measurment_directory}/{dataset}/Pos0/img_000000000_Default_000.tif",
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}/raw.npy",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/read_tiff.tsv",
    run:
        data = read_tiff_img(file_path=input.data, border_px=0)
        Path(output[0]).parent.mkdir(exist_ok=True, parents=True)
        np.save(file=output.data, arr=data)

rule copy_metadata:
    input:
        data="data/raw/foils/{measurment_directory}/{dataset}/Pos0/metadata.txt",
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}/metadata.txt",
    run:
        Path(output[0]).parent.mkdir(exist_ok=True, parents=True)
        with open(input.data, "r") as f:
            metadata = json.load(f)
            with open(output.data, "w") as f:
                json.dump(metadata, f)

rule background_constant_subtraction:
    input:
        data="data/interim/foils/{measurment_directory}/{dataset}lv/raw.npy",
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}lv/raw-bg-const-removed.npy",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}lv/benchmark/background_constant_subtraction.tsv",
    run:
        data_signal = np.load(file=input.data)
        constant_background_level = np.nanmin(data_signal)
        data_bg_removed = subtract_background(input=data_signal, const_bg=constant_background_level)
        np.save(file=output.data, arr=data_bg_removed)

rule image_contour:
    input:
        data=rules.background_constant_subtraction.output.data,
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}lv/raw-threshold.npy",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}lv/benchmark/image_contour.tsv",
    run:
        data_lv = np.load(file=input.data)
        data_thres = img_for_circle_detection(input=data_lv)
        np.save(file=output.data, arr=data_thres)

rule detector_circle:
    input:
        data=rules.image_contour.output.data,
    output:
        det_circle="data/interim/foils/{measurment_directory}/{dataset}lv/det-circle.json",
        aligned_det_circle="data/interim/foils/{measurment_directory}/{dataset}lv/aligned-det-circle.json",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/detector_circle.tsv",
    run:
        data = np.load(file=input.data)
        det_circle = find_circle_hough_method(data)
        aligned_det_circle = Circle(x=data.shape[1]/2, y=data.shape[0]/2, r=det_circle.r)
        det_circle.save_json(output.det_circle)
        aligned_det_circle.save_json(output.aligned_det_circle)

rule circles:
    input:
        data=rules.image_contour.output.data,
        detector_circle=rules.detector_circle.output.det_circle
    output:
        analysis_circle="data/interim/foils/{measurment_directory}/{dataset}/analysis-circle.json",
        aligned_analysis_circle="data/interim/foils/{measurment_directory}/{dataset}/aligned-analysis-circle.json"
    params:
        analysis_radius=analysis_radius
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/analysis_circles.tsv"
    run:
        data = np.load(file=input.data)
        detector_circle = Circle.from_json(input.detector_circle)

        analysis_circle = Circle(x=detector_circle.x, y=detector_circle.y, r=params.analysis_radius)
        aligned_analysis_circle = Circle(x=data.shape[1]/2, y=data.shape[0]/2, r=params.analysis_radius)

        analysis_circle.save_json(output.analysis_circle)
        aligned_analysis_circle.save_json(output.aligned_analysis_circle)

rule signal_on_circle:
    input:
        image="data/interim/foils/{measurment_directory}/{dataset}lv/raw.npy",
        circle=rules.detector_circle.output.det_circle,
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}lv/angle.npy",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/signal_on_circle.tsv",
    run:
        data = np.load(file=input.image)
        circle = Circle.from_json(input.circle)
        meas_radius = max(0, circle.r - 60)  # radius cannot be negative !
        meas_circle = Circle(circle.x, circle.y, meas_radius)
        array_of_angles_deg, values_on_circle = get_line_circle(data, circ=meas_circle, step_angle_deg=0.1)
        values_on_circle[values_on_circle == 0.0] = np.nan
        angle_with_min_value, _ = get_angle_with_min_value(array_of_angles_deg, values_on_circle, median_filter_size=10)
        np.save(file=output.data, arr=np.array(angle_with_min_value))

