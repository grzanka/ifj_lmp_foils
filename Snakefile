wildcard_constraints:
    measurment_directory="\w+"

configfile: "data/external/snakemake_configs/main.yaml"
datasets = config["datasets"]
background = config["background"]
measurement_directory = config["measurment_directory"]
analysis_radius = config["analysis_radius"]

ff_white_image = '2022_08_22_flat_field/FF_2sLED_U340/FF_1'
ff_radius = config['ff_radius']

vmax_for_plotting = config['vmax_for_plotting']

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
            "data/interim/foils/{measurment_directory}/images2d.pdf",
            zip,
            datasets=datasets,
            measurment_directory=[measurement_directory]*len(datasets)
        ),
        # expand(
        #     "data/interim/foils/{measurment_directory}/signal.pdf",
        #     zip,
        #     datasets=datasets,
        #     measurment_directory=[measurement_directory]*len(datasets)
        # )


include: "rules/raw.smk"


rule plot_all_2d_images:
    input:
        plot_file=expand("data/interim/foils/{measurment_directory}/{datasets}/images2d.pdf", 
            zip,
            datasets=datasets,
            measurment_directory=[measurement_directory]*len(datasets))
    output:
        plot_file="data/interim/foils/{measurment_directory}/images2d.pdf",
    shell:
        "pdftk {input.plot_file} cat output {output.plot_file}"

# rule plot_signal_summary:
#     input:
#         raw_files=expand(
#             "data/interim/foils/{measurment_directory}/{datasets}/raw.npy",
#             zip,
#             datasets=datasets,
#             measurment_directory=[measurement_directory]*len(datasets)
#         ),
#     output:
#         plot_file="data/interim/foils/{measurment_directory}/signal.pdf",
#     shell:
#         df_data = defaultdict(list)
#         df_data['det_id'] = []
#         df_data['timestamp'] = []
#         for filename in ('raw', 'raw-bg-image-removed', 'raw-aligned', 'raw-after-ff'):
#             df_data[f'{filename}_signal_mean'] = []
#             df_data[f'{filename}_signal_std'] = []
#         for det in list_of_datasets:
#             df_data['det_id'].append(int(det))
#             df_data['timestamp'].append(get_timestamp(f'{raw_path}/{dataset}/{det}/Pos0/metadata.txt'))

#             analysis_circle = Circle.from_json(f'{interim_path}/{dataset}/{det}/analysis-circle.json')
#             aligned_analysis_circle = Circle.from_json(f'{interim_path}/{dataset}/{det}/aligned-analysis-circle.json')

#             for filename in ('raw', 'raw-bg-image-removed', 'raw-aligned', 'raw-after-ff'):
#                 data = np.load(Path(dataset_path, det, f'{filename}.npy'))
#                 circle = analysis_circle
#                 if 'aligned' in filename:
#                     circle = aligned_analysis_circle
#                 mean, std = get_mean_std(data, circle)
#                 df_data[f'{filename}_signal_mean'].append(mean)
#                 df_data[f'{filename}_signal_std'].append(std)
#         df = pd.DataFrame.from_dict(df_data)
#         df.head()