LMP foils analysis workflow
===========================

Datasets:
---------

2020.10.09 - BP or SOBP eyeball irradiation
-------------------------------------------

entrance dose 20 Gy
collimator with 10mm diameter


2020.10.09 - SOBP eyeball
-------------------------

Processing time 7min on 16 CPU

```bash
time snakemake --configfile data/external/snakemake_configs/TODO.yaml --cores all
```

2020.10.12 - Co60 reference
---------------------------

60 Gy reference dose

```bash
time snakemake --configfile data/external/snakemake_configs/2020_10_12_Co60.yaml --cores all
```

2021.11.18 - BP/SOBP
--------------------

Processing time 4min on 16 CPU

```bash
snakemake --snakefile Snakefile_2021_11_18_bp --cores all
```

```bash
snakemake --snakefile Snakefile_2021_11_18_sobp --cores all
```

2022.05.25 - BP/SOBP
--------------------

BP/SOBP comparison for 3 (for each) detectors stacked together, irradiated in the plateau with 60 Gy for both BP/SOBP

TODO

2022.06.09 - BP/SOBP
--------------------

Efficiency at the distal part of BP and SOBP

TODO

2022.11.03 - Co60 reference
---------------------------

- 60 Gy reference dose
- missing data for det 12
- missing live view image for det 2

```bash
snakemake --snakefile Snakefile_2022_11_03_Co60 --cores all
```

2022.11.17 - BP irradiation
---------------------------

- 12 Gy reference dose

TODO: check which radius is used for reference radiation
what happens if a different radius is for reference than for protons

```bash
time snakemake --configfile data/external/snakemake_configs/2022_11_17_bp.yaml --cores all
time snakemake --configfile data/external/snakemake_configs/2022_11_18_sobp.yaml --cores all
```

Helper
------

```bash
find . -name "*_lv" -exec bash -c 'mv $1 ${1/\_/}' bash {} \;
```


fetching files from OneDrive:

```bash
rclone sync onedriveifj:LMP_foils/raw data/raw/foils
```

submitting interim results:
```bash
rclone sync data/interim/foils ifjonedrive_interim:
```


run all datasets:
```bash
find data/external/snakemake_configs/ -name "20*yaml" -exec snakemake --configfile {} --cores all \;
```