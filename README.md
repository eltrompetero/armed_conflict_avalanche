# Construction and Analysis of Conflict Avalanches

This repository contains code for the construction and analysis of conflict avalanches, including simulation of the model and incorporation of social data.

## Modules

- **Cluster:** This module contains routines for generating Voronoi cells and clustering conflict events with them.
- **Avalanches:** The `Avalanches` module is used for the simulation of the fractal conflict model.
- **pipeline.py:** This file contains routines for the final analysis.

## Installation

To run the notebook, please follow these steps:

1. Clone the private repository:

    ```
    $ git clone git@github.com:eltrompetero/armed_conflict_avalanche.git
    ```

2. Move the current working directory inside the repository:

    ```
    $ cd armed_conflict_avalanche
    ```

3. Install custom repositories into your current working directory:

    ```
    $ git clone https://github.com/eltrompetero/misc.git
    $ git clone https://github.com/eltrompetero/workspace.git
    ```

4. Install the custom repository "Voronoi_globe" as per the instructions given at [https://github.com/eltrompetero/voronoi_globe](https://github.com/eltrompetero/voronoi_globe).

5. Install a custom Anaconda virtual environment using the spec file provided in the repository. To do that, inside the working directory (which contains the spec file), run the following command:

    ```
    $ conda create --name <ENVIRONMENT NAME> --file specfile_armed_conflict.txt
    ```

6. If some packages are missing, it is easiest to install them from the conda-forge channel. For example, if `geopandas` is missing, run the following command:

    ```
    $ conda install -n arco -c conda-forge geopandas
    ```

7. Copy the `voronoi_grids` and `data` folders to the working directory. You can download these folders from the Dropbox link.

8. In some code, you must specify the current working directory in the variable `wd` or its analog.
