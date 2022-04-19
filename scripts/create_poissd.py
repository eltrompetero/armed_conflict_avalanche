# ====================================================================================== #
# Code for producing a random Voronoi cell grid across the African continent.
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import sys
from workspace.utils import load_pickle, save_pickle

from arcolanche import *

WIDTH = (0.05235987755982988, 1.6406094968746698)
HEIGHT = (-0.7853981633974483, 0.8203047484373349)


def create_one_grid(gridix, dx):
    assert gridix>=0
    assert dx in [40,80,160,320,640,1280]

    if dx==40:
        # generate centers of voronoi cells
        poissd = PoissonDiscSphere(np.pi/dx,
                                   width_bds=WIDTH,
                                   height_bds=HEIGHT)

    else:
        # load coarse grid that will be used to group cells in finer grid
        poissd = pickle.load(open(f'voronoi_grids/{dx//2}/{str(gridix).zfill(2)}.p','rb'))['poissd']
        coarsegrid = poissd.samples

        # generate centers of voronoi cells
        poissd = PoissonDiscSphere(np.pi/dx,
                                   width_bds=WIDTH,
                                   height_bds=HEIGHT,
                                   coarse_grid=coarsegrid,
                                   k_coarse=15)
    poissd.sample()

    fname = f'voronoi_grids/{dx}/{str(gridix).zfill(2)}.p'
    if os.path.isfile(fname):
        warn("Overwriting existing file.")
    save_pickle(['poissd'], fname, True)

def main(gridix):
    """Loop thru Voronoi tessellations starting with coarsest grid of solid angle
    pi/40, which then forms the basis for a coarse grid to speed up finer grids in
    nested chain.
    """

    for dx in [40,80,160,320,640,1280]:
        create_one_grid(gridix, dx)
        print(f"Done with grid {gridix} and dx {dx}.\n")

if __name__=='__main__':
    gridix = int(sys.argv[1])
    
    main(gridix)

