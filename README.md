[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12157646.svg)](https://doi.org/10.5281/zenodo.12157646)

# DelVG meta study
This is the code for the analyses performed in the publication ().

## setup
Before running the with different datasets the repository needs to be set up accordingly.

1. change the variables in `utils.py` line 16 & 17:
   ```
    DATAPATH = "/path/to/datasets"
    RESULTSPATH = "/path/to/resultfolder"
   ```
2. Put your data in the folder that is defined in DATAPATH. Each each publication should get a new folder and in this folder each SRA entry is a single .csv file. To generate datasets the [DI identification pipeline](https://github.com/BROOKELAB/Influenza-virus-DI-identification-pipeline) by Alnaji et al. can be used.
3. Install all dependencies. This can be done by conda using the following command:
   ```
   conda env create --file=env.yml
   ```
