# Armed conflict avalanches

Code for construction conflict avalanches and their analysis including simulation of model
and incorporation of social data.

Cluster module contains routines for generating Voronoi cells and clustering conflict
events with them. Avalanches module is for simulation of theoretical model.

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
scientific python packages). As a shortcut, a spec list is available in the GitHub repo
for Anaconda.
```bash
$ conda create --name armed_conflict --file pyutils/spec-file.txt
$ conda activate armed_conflict
```

