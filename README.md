# ARmed COnflict avaLANCHEs (arcolanche)

Code for construction and analysis of conflict avalanches including simulation of model
and incorporation of social data.

- Cluster module contains routines for generating Voronoi cells and clustering conflict
events with them.
- Avalanches module is for simulation of fractal conflict model.
- pipeline.py contains routines for final analysis.


# Installation
In order to run the notebook, you will need to take several steps.
1. Install the private repo.
    ```bash
    $ git clone git@github.com:eltrompetero/armed_conflict_avalanche.git
    ```
2. Move the current working directory inside the repo.
    ```bash
    $ cd armed_conflict_avalanche
    ```
3. Install custom repositories into your current working directory.
    ```bash
    $ git clone https://github.com/eltrompetero/misc.git
    $ git clone https://github.com/eltrompetero/workspace.git
    ```
2. Install an appropriate Python environment. Best course of action is to create a virtual environment in order to separate this environment from others and to preserve it. I prefer to do this with Anaconda. As a shortcut, a YML file is available in the GitHub repo.
    ```bash
    $ conda env create -f arcolanche/arco.yml
    $ conda activate arco
    ```
    If some packages are missing, it is easiest to install them from them the conda-forge channel. For example, if geopandas is missing, one would write
    ```bash
    $ conda install -n arco -c conda-forge geopandas
    ```
3. Symlink the necessary data files into the current working directory.
    ```bash
    $ ln -s /fastcache/armed_conflict/voronoi_grids voronoi_grids
    $ ln -s /fastcache/armed_conflict/africa_roads africa_roads
    $ ln -s /fastcache/armed_conflict/population_af population_af
    ```
4. In some code, you must specify the current working directory in the variable `wd` or its analog.

# Scripts
- For creating a nested sequence of Voronoi cells with script. Last integer argument is the Voronoi cell index. Existing files are overwritten.
    ```bash
    cp scripts/create_poissd.py ./ && python create_poissd.py 0
    ```

