# ARmed COnflict avaLANCHEs (arcolanche)

Code for construction and analysis of conflict avalanches including simulation of model
and incorporation of social data.

- Cluster module contains routines for generating Voronoi cells and clustering conflict
events with them.
- Avalanches module is for simulation of fractal conflict model.
- pipeline.py contains routines for final analysis.


# Installation
In order to run the notebook, you will need to install custom repositories into path.
```bash
$ git clone https://github.com/eltrompetero/misc.git
$ git clone https://github.com/eltrompetero/workspace.git
```
Then the armed conflict repo needs to be on your path. From your working directory,
you should run
```bash
$ git clone https://github.com/eltrompetero/armed_conflict.git
```

You will also need an environment with geopandas (also pandas, cartopy, and standard
scientific python packages). As a shortcut, a YML and spec list is available in the GitHub repo
for Anaconda.

If you're running Linux, then you can use
```bash
$ conda create --name arco --file arcolanche/spec-file.txt
$ conda activate arco
```
Otherwise
```bash
$ conda env create -f arcolanche/arco.yml
$ conda activate arco
```


# Running
Creating a nested sequence of Voronoi cells with script. Last integer argument is the
Voronoi cell index. Existing files are overwritten.
```bash
cp scripts/create_poissd.py ./ && python create_poissd.py 0
```

