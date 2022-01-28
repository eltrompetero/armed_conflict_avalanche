# ====================================================================================== #
# Code for producing a random Voronoi cell grid across the African subcontinent.
# Author : Eddie Lee, edlee@csh.ac.at
# 2020-12-07
# ====================================================================================== #
from pyutils import *
from workspace.utils import load_pickle, save_pickle

gridix = 1
dx = 2560

# load coarse grid that will be used to group cells in finer grid
poissd = pickle.load(open(f'voronoi_grids/{dx//2}/{str(gridix).zfill(2)}.p','rb'))['poissd']
width = poissd.width
height = poissd.height
coarsegrid = poissd.samples

# generate centers of voronoi cells
poissd = PoissonDiscSphere(np.pi/dx,
                           width_bds=width,
                           height_bds=height,
                           coarse_grid=coarsegrid,
                           k_coarse=15)

save_pickle(['poissd'], f'voronoi_grids/{dx}/{str(gridix).zfill(2)}.p', True)

