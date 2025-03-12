[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14872995.svg)](https://doi.org/10.5281/zenodo.14872995)

# DelVG meta study
This is the code for the analyses performed in the publication "Meta-analysis of genomic characteristics for antiviral influenza defective interfering particle prioritization".

## Setup
Before running the with different datasets the repository needs to be set up accordingly.

1. change the variables in `utils.py` line 16 & 17:
   ```
    DATAPATH = "/path/to/datasets"
    RESULTSPATH = "/path/to/resultfolder"
   ```
2. The necesary datasets are already available in the folder 'data'. So you can set the DATAPATH to this directory (unless you want to use your own data)
3. Download and store the fasta files of the strains in a folder called `strain_segment_fastas`. Each strain is an independent folder with the segments being individual fasta files.
4. Install all dependencies. This can be done by conda using the following command:
   ```
   conda env create --file=env.yml
   ```

## Running analyses
To run all analyses the script src/run_analysis_scripts.sh is available.
First you need to activate the conda environment.
   ```
   conda activate dips
   ```
Afterwards move into the folder with the code (assuming you are in the root of this repo):
   ```
   cd src  
   ```
Then it can be run using the following command:
   ```
   bash run_analysis_scripts.sh
   ```

For some analyses we used jupyter notebooks for easier accessibility.
These need to be run separately.
