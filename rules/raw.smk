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

rule background_image_subtraction:
    input:
        signal="data/interim/foils/{measurment_directory}/{dataset}/raw.npy",
        background="data/interim/foils/{measurment_directory}/" +f"{background}/raw.npy"
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}/raw-bg-image-removed.npy",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/background_image_subtraction.tsv",
    run:
        data_signal = np.load(file=input.signal)
        data_background = np.load(file=input.background)
        data_bg_removed = subtract_background(input=data_signal, img_bg=data_background)
        np.save(file=output.data, arr=data_bg_removed)

rule background_constant_subtraction:
    input:
        data="data/interim/foils/{measurment_directory}/{dataset}lv/raw.npy",
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}lv/raw-bg-const-removed.npy",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/background_constant_subtraction.tsv",
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
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/image_contour.tsv",
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


rule flat_field:
    input:
        data="data/interim/foils/{measurment_directory}/{dataset}/raw-bg-image-removed.npy",
        white_image=f"data/interim/foils/{ff_white_image}/raw.npy"
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}/raw-after-ff.npy",
        ff_circle="data/interim/foils/{measurment_directory}/{dataset}/ff-circle.json",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/flat_field.tsv",
    params:
        radius=ff_radius,
    run:
        data = np.load(file=input.data)
        white_data = np.load(file=input.white_image)

        # circle around the image center with given radius
        ff_circle = Circle(x=data.shape[1]/2, y=data.shape[0]/2, r=params.radius)
        mask = create_circular_mask(img=white_data, circle_px=ff_circle)
        
        # calculate FF correction on the full image
        # mean value is calculated only in the given radius from detector center
        gain_full = np.nanmean(white_data[mask]) / white_data
        corr_data_full = data * gain_full

        ff_circle.save_json(output.ff_circle)
        np.save(file=output.data, arr=corr_data_full)

rule signal_on_circle:
    input:
        image="data/interim/foils/{measurment_directory}/{dataset}lv/raw.npy",
        circle=rules.detector_circle.output.det_circle,
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}/angle.npy",
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

rule align_top:
    input:
        angle=rules.signal_on_circle.output.data,
        image=rules.flat_field.output.data,
        circle=rules.detector_circle.output.det_circle
    output:
        data="data/interim/foils/{measurment_directory}/{dataset}/raw-aligned.npy",
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/align_top.tsv",
    run:
        angle = np.load(file=input.angle)
        data = np.load(file=input.image)
        circle = Circle.from_json(input.circle)         # detector circle

        # shift image, so the image is aligned with detector center
        shifted = ndimage.shift(data, (data.shape[1]/2-circle.y,data.shape[0]/2-circle.x), cval=np.nan, prefilter=False)
        # rotate image, so the mark is on the top
        rotated = ndimage.rotate(shifted, -angle, cval=np.nan, reshape=False, prefilter=False)

        np.save(file=output.data, arr=np.array(rotated))

rule plot_2d_images:
    input:
        bg_removed=rules.background_image_subtraction.output.data,
        flat_field=rules.flat_field.output.data,
        aligned=rules.align_top.output.data,
        det_circle=rules.detector_circle.output.det_circle,
        aligned_det_circle=rules.detector_circle.output.aligned_det_circle,
        analysis_circle=rules.circles.output.analysis_circle,
        aligned_analysis_circle=rules.circles.output.aligned_analysis_circle,
    output:
        plot_file="data/interim/foils/{measurment_directory}/{dataset}/images2d.pdf",
    params:
        vmax_for_plotting=vmax_for_plotting,
    benchmark:
        "data/interim/foils/{measurment_directory}/{dataset}/benchmark/plot_stages.tsv",
    run:
        det_circle = Circle.from_json(input.det_circle)
        aligned_det_circle = Circle.from_json(input.aligned_det_circle)
        analysis_circle = Circle.from_json(input.analysis_circle)
        aligned_analysis_circle = Circle.from_json(input.aligned_analysis_circle)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,10))

        for col_id, filename in enumerate((input.bg_removed, input.flat_field, input.aligned)):        
            ref_data = np.load(file=filename)
            ref_data_for_plotting = np.clip(ref_data, a_min=None, a_max=np.nanpercentile(a=ref_data, q=95))
            ref_data_plot = axes[col_id].imshow(ref_data_for_plotting, cmap='terrain', vmin=0, vmax=vmax_for_plotting);
            basename = Path(filename).stem

            axes[col_id].set_title(f'{basename}')

            if 'aligned' in basename:
                axes[col_id].add_artist(plt.Circle(xy=(aligned_det_circle.x, aligned_det_circle.y), radius=aligned_det_circle.r, color='black', fill=False, transform=axes[col_id].transData))
                axes[col_id].add_artist(plt.Circle(xy=(aligned_analysis_circle.x, aligned_analysis_circle.y), radius=aligned_analysis_circle.r, color='red', fill=False, transform=axes[col_id].transData))
            else:
                axes[col_id].add_artist(plt.Circle(xy=(det_circle.x, det_circle.y), radius=det_circle.r, color='black', fill=False, transform=axes[col_id].transData))
                axes[col_id].add_artist(plt.Circle(xy=(analysis_circle.x, analysis_circle.y), radius=analysis_circle.r, color='red', fill=False, transform=axes[col_id].transData))


        output_path = Path(output.plot_file)
        det_id = output_path.parent.name
        meas_id = output_path.parent.parent.name
        fig.colorbar(ref_data_plot, ax=axes, location='right', shrink=0.4)
        fig.suptitle(f"Measurement {meas_id}, dataset {det_id}", y=0.75)
        fig.savefig(output.plot_file, bbox_inches='tight')
