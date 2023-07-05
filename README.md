# ARmed COnflict avaLANCHEs (arcolanche)

![](https://github.com/eltrompetero/armed_conflict_avalanche/blob/PNAS_Nexus_2023/avalanches_dt32_dx320_gridix3.gif)

This repository contains code for the construction and analysis of conflict avalanches, including simulation of the model and incorporation of social data.

## Modules

- **Cluster:** This module contains routines for generating Voronoi cells and clustering conflict events with them.
- **Avalanches:** The `Avalanches` module is used for the simulation of the fractal conflict model.
- **pipeline.py:** This file contains routines for the final analysis.

## Installation

To run the notebook, please follow these steps:

1. Clone this specific branch of the repository:

    ```
    $ git clone --branch PNAS_Nexus_2023 https://github.com/eltrompetero/armed_conflict_avalanche.git
    ```

2. Move inside the repository which will be our working directory:

    ```
    $ cd armed_conflict_avalanche
    ```

3. Install custom repositories into your current working directory:

    ```
    $ git clone https://github.com/eltrompetero/misc.git
    $ git clone https://github.com/eltrompetero/workspace.git
    ```

4. Install the custom repository `Voronoi_globe` as per the instructions given at [https://github.com/eltrompetero/voronoi_globe](https://github.com/eltrompetero/voronoi_globe).

5. Install a custom Anaconda virtual environment using the spec file provided in the repository. To do that, inside the working directory (which contains the spec file), run the following command:

    ```
    $ conda create --name <ENVIRONMENT NAME> --file specfile_armed_conflict.txt
    ```
6. Install `vincenty` and `mycolorpy` using pip in the conda environment since they are unavailable in conda-forge.
   ```
    $ python3 -m pip install vincenty
    $ python3 -m pip install mycolorpy
   ```
   
8. If some packages are missing, it is easiest to install them from the conda-forge channel. For example, if `geopandas` is missing, run the following command:

    ```
    $ conda install -n <ENVIRONMENT NAME> -c conda-forge geopandas
    ```

9. Copy the `voronoi_grids` and `data` folders to the working directory. You can download these folders from the Zenodo link. If you wish to generate new voronoi grids, follow the instructions at [voronoi_globe](https://github.com/eltrompetero/voronoi_globe).

## Testing the installation

1. Inside the working directory, activate the custom conda environment:
    ```
    $ conda activate <ENVIRONMENT NAME>
    ```
2. Run `installation_test.ipynb` and go through the instructions.

## Reproducing figures used in the paper

1. Inside the working directory, activate the custom conda environment.
2. Run `paper_pipeline.ipynb` and go through the instructions.

