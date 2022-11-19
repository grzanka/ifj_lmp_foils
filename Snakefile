import numpy as np

wildcard_constraints:
    measurment_directory="\w+"


# IRI datasets
configfile: "data/external/snakemake_configs/main.yaml"
datasets = config["datasets"]
background = config["background"]
measurement_directory = config["measurment_directory"]
analysis_radius = config["analysis_radius"]

# basic datasets
ff_white_image = '2022_08_22_flat_field/FF_2sLED_U340/FF_1'
ff_radius = config['ff_radius']
vmax_ref_data = 50

# master rule which requires generates all plots
rule all:
    input:
        # expect raw file readed for basic, IRI and FF datasets
        expand(
            "data/interim/foils/{measurment_directory}/{datasets}/raw.npy",
            zip,
            datasets=datasets,
            measurment_directory=[measurement_directory]*len(datasets)
        ),
        expand(
            "data/interim/foils/{measurment_directory}/{datasets}/raw-after-ff.npy",
            zip,
            datasets=datasets,
            measurment_directory=[measurement_directory]*len(datasets)
        ),
        expand(
            "data/interim/foils/{measurment_directory}/{datasets}/raw-aligned.npy",
            zip,
            datasets=datasets,
            measurment_directory=[measurement_directory]*len(datasets)
        ),
        expand(
            "data/interim/foils/{measurment_directory}/{datasets}/stages.pdf",
            zip,
            datasets=datasets,
            measurment_directory=[measurement_directory]*len(datasets)
        ),
        expand(
            "data/interim/foils/{measurment_directory}/stages.pdf",
            zip,
            datasets=datasets,
            measurment_directory=[measurement_directory]*len(datasets)
        )


include: "rules/raw.smk"


rule plot_all_stages:
    input:
        plot_file=expand("data/interim/foils/{measurment_directory}/{datasets}/stages.pdf", 
            zip,
            datasets=datasets,
            measurment_directory=[measurement_directory]*len(datasets))
    output:
        plot_file="data/interim/foils/{measurment_directory}/stages.pdf",
    shell:
        "pdftk {input.plot_file} cat output {output.plot_file}"