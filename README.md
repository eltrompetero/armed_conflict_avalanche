# ARmed COnflict avaLANCHEs (arcolanche)

<p align="center">
  <img src="https://github.com/eltrompetero/armed_conflict_avalanche/blob/PNAS_Nexus_2023/avalanches_dt32_dx320_gridix3.gif" />
</p>

This repository contains code for the construction and analysis of conflict avalanches, including simulation of the model and incorporation of social data.

If you use this Python package for your research, please consider citing the following paper:

```
Kushwaha, Niraj, and Edward D. Lee. "Discovering the mesoscale for chains of conflict." arXiv preprint arXiv:2212.02652 (2022).
```


## Important modules

- **construct.py:** This module is used to construct conflict avalanches.
- **network.py:** This module is used to construct causal network using transfer entropy.
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
    $ conda activate <ENVIRONMENT NAME>
    $ python3 -m pip install vincenty
    $ python3 -m pip install mycolorpy
   ```
   
8. If some packages are missing, it is easiest to install them from the conda-forge channel. For example, if `geopandas` is missing, run the following command:

    ```
    $ conda install -n <ENVIRONMENT NAME> -c conda-forge geopandas
    ```

9. Copy the `voronoi_grids` folder to the working directory. You can download this folder from [Zenodo](https://doi.org/10.5281/zenodo.8117567). If you wish to generate new voronoi grids, follow the instructions at [Voronoi_globe](https://github.com/eltrompetero/voronoi_globe).

10. Create a new folder called `data` and add your ACLED dataset to this folder. You can download the ACLED dataset from [here](https://acleddata.com/data-export-tool/). Rename the datafile to `ACLED_data.csv`. We used the following parameters to download the dataset from ACLED (we downloaded the dataset on 30th September 2022): 
    ```
    From : 01/01/1997
    To : 31/12/2019
    Event Type : All
    Sub Event : All
    Actor Type : All
    Actor : All
    Region : Eastern Africa, Middle Africa, Northern Africa, Southern Africa, Western Africa
    Country : All
    Location : All
    ```
You can use these filters to get the same dataset as us.

Note: If you want to use this code to analyse regions other than Africa, you need to first generate voronoi grids for that region (Use [Voronoi_globe](https://github.com/eltrompetero/voronoi_globe)).

<!--
## Testing the installation

1. Inside the working directory, activate the custom conda environment:
    ```
    $ conda activate <ENVIRONMENT NAME>
    ```
2. Run `installation_test.ipynb` and go through the instructions.
-->

## Reproducing figures used in the paper

1. Inside the working directory, activate the custom conda environment.
    ```
    $ conda activate <ENVIRONMENT NAME>
    ```
2. Run `paper_pipeline.ipynb` and go through the instructions.


## Questions and Feedback

If you have any questions about how to utilize specific features of our Python package or if you would like to suggest improvements to the code, we encourage you to reach out. We value your feedback and are committed to enhancing your experience with our package.

Please feel free to use the discussions and issues section of github for any inquiries or suggestions you may have. We appreciate your contribution to making our package even better.

Thank you for using our Python package!
