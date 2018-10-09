from .acled_utils import *
from data_sets.acled import *
import pickle


def _vtess_loop(args):
    """
    Loop through increasingly finer resolutions. 

    Parameters
    ----------
    args : list
        (p, fname, initial_coarse_grid)
        p : list of denominators to divide pi by. these should be increasing (finer resolution)
        fname : name under which to save results (folder is given by p) and excluding '.p' suffix
        initial_coarse_grid : optional coarse grid to use for first sampling run
    """
    from numpy import pi
    if len(args)==2:
        p, fname=args
        initial_coarse_grid=None
    elif len(args)==3:
        p, fname, initial_coarse_grid=args
    else:
        raise Exception
    
    coarseGrid=initial_coarse_grid
    for p_ in p:
        poissd=PoissonDiscSphere(pi/p_,
                                 coarse_grid=coarseGrid,
                                 width_bds=(3/180*pi, 94/180*pi),
                                 height_bds=(-45/180*pi, 47/180*pi))
        poissd.sample()
        coarseGrid=poissd.samples
    
        pickle.dump({'poissd':poissd}, open('voronoi_grids/%d/%s.p'%(p_, fname), 'wb'), -1)
        print('Done with files %s resolution %d'%(fname,p_))
    
def voronoi(piDenom, nStarts, fileno_start=0, n_jobs=None, initial_coarse_grid=None):
    """Run voronoi tessellation. Results are saved into the voronoi_grids folder. Chain coarse
    gridding such that results of a coarser grid are fed into the finer grid.
    
    Parameters
    ----------
    piDenom : ndarray
        pi is divided by this number to determine the spacing of the Voronoi tessellation
    nStarts : int
    fileno_start : int,0
        Start file name counts from this number.
    """
    import multiprocess as mp
    n_jobs=n_jobs or mp.cpu_count()-1
    assert (np.diff(piDenom)>0).all(), "Grid must get finer."

    # set up directories
    for p in piDenom:
        if not os.path.isdir('voronoi_grids/%d'%p):
            os.mkdir('voronoi_grids/%d'%p)
            
    # setup up thread arguments
    args=[]
    if initial_coarse_grid is None:
        for i in range(nStarts):
            args.append( (piDenom, str(i+fileno_start).zfill(2)) )
    else:
        for i in range(nStarts):
            args.append( (piDenom, str(i+fileno_start).zfill(2), initial_coarse_grid) )

    pool=mp.Pool(n_jobs)
    pool.map(_vtess_loop, args)
    pool.close()

def extract_info_voronoi(subdf, dx, dt, grid_no,
                         this_layer_pixel=None,
                         next_layer_pixel=None,
                         split_geo=True,
                         split_actor=True,
                         split_date=True):
    """
    Parameters
    ----------
    subdf : pandas.Dataframe
        To split.
    dx : int
        Denominator for pi.
    dt : int
        Number of days to threshold.
    grid_no : int
        Saved tessellation to use. These are contained in voronoi_grids/.
    split_geo : bool,True
    split_actor : bool,True
    split_date : bool,True
        
    Returns
    -------
    geoSplit : list
    eventCount
    duration
    fatalities
    diameter
    """

    assert split_geo or split_actor or split_date

    subdf=subdf.loc[:,('EVENT_DATE','LATITUDE','LONGITUDE','FATALITIES','actors')]
     
    # Split by voronoi tiles.
    if split_geo:
        if not next_layer_pixel is None:
            assert len(this_layer_pixel)==len(subdf)
            geoSplit_=[]
    
            # map onto next layer
            for ix in np.unique(next_layer_pixel):
                eventsToAdd=[]
                matchingPixelsInThisLayer=np.where(ix==next_layer_pixel)[0]
                for ixx in matchingPixelsInThisLayer:
                    eventsToAdd.append(subdf.iloc[this_layer_pixel==ixx])

                geoSplit_.append(pd.concat(eventsToAdd).sort_values('EVENT_DATE', axis=0))

        else:
            # do lowest level tiling
            poissd=pickle.load(open('voronoi_grids/%d/%s.p'%(dx, str(grid_no).zfill(2)), 'rb'))['poissd']
            # an offset of 330 degrees center the tessellation above Africa
            geoSplit_, pixIx=pixelate_voronoi(subdf, poissd, 330)
    else:
        geoSplit_=[subdf]
    
    # Split by actors.
    if split_actor:
        geoSplit_=[split_by_actor(s)[0] for s in geoSplit_]
        geoSplit_=list(chain.from_iterable(geoSplit_))
    elif not split_geo:
        geoSplit_=[subdf]

    # Split by date.
    if split_date:
        geoSplit=[]
        for g in geoSplit_:
            g_=tpixelate(g, dt, subdf['EVENT_DATE'].min(), subdf['EVENT_DATE'].max()) 
            geoSplit+=g_
            l1, l2=sum([sum([len(j) for j in i]) for i in g_]), len(g)
            assert l1==l2,(l1,l2)
    
    try:
        geoSplit;
    except NameError:
        geoSplit=geoSplit_
    
    eventCount, duration, fatalities, diameter=duration_and_size(geoSplit)
    try:
        return geoSplit, eventCount, duration, fatalities, diameter, pixIx
    except:
        return geoSplit, eventCount, duration, fatalities, diameter

def extract_info(subdf,dx,dt,
                 add_random_orientation=False,
                 split_geo=True,
                 split_actor=True,
                 split_date=True):
    """
    Parameters
    ----------
    subdf : pandas.Dataframe
        To split.
    dx : int
        Power of 2**n pixels to use. Special value of -1 is no pixelation.
    dt : int
        Number of days to threshold.
    add_random_orientation : bool,False
        Rotate events randomly around the globe (this is for reducing the effect of a 
        particular grid chosen).
    split_geo : bool,True
    split_actor : bool,True
    split_date : bool,True
        
    Returns
    -------
    geoSplit : list
    eventCount
    duration
    fatalities
    diameter
    """
    assert split_geo or split_actor or split_date

    subdf=subdf.loc[:,('EVENT_DATE','LATITUDE','LONGITUDE','FATALITIES','actors')]
    if add_random_orientation:
        randlat,randlon=add_random_orientation
        subdf.loc[:,'LATITUDE']=((subdf.loc[:,'LATITUDE']+randlat)%180)-90
        subdf.loc[:,'LONGITUDE']=((subdf.loc[:,'LONGITUDE']+randlon)%360)-180
    
    # Split by healpy pixels
    if split_geo:
        if dx==-1:
            geoSplit_, pixelDiameter=[subdf], 6370*np.pi
        else:
            geoSplit_, pixelDiameter=pixelate(subdf, 2**dx)
    else:
        geoSplit_=[np.arange(len(subdf), dtype=int)]
    
    # Split by actors.
    if split_actor:
        geoSplit_=[[s[ix] for ix in split_by_actor(subdf.iloc[s.tolist(),:])[0]] for s in geoSplit_]
        geoSplit_=list(chain.from_iterable(geoSplit_))
    elif not split_geo:
        geoSplit_=[np.arange(len(subdf), dtype=int)]

    # Split by date.
    if split_date:
        geoSplit=[]
        for g in geoSplit_:
            g_=tpixelate(subdf.iloc[g], dt, subdf['EVENT_DATE'].min(), subdf['EVENT_DATE'].max()) 
            geoSplit+=[g[i] for i in g_]
            l1, l2=sum([len(i) for i in g_]), len(g)
            assert l1==l2, (l1,l2)
    
    try:
        geoSplit;
    except NameError:
        geoSplit=geoSplit_
    
    eventCount, duration, fatalities, diameter=duration_and_size([subdf.iloc[ix] for ix in geoSplit])
    return geoSplit,eventCount,duration,fatalities,diameter

def setup_quickload(path):
    """Since the pickles contain geoSplit which is a large variable, write quickload files 
    that allow you to avoid loading that. Iterate through all files in the given directory assuming
    that they're pickles.
    
    Parameters
    ----------
    path : str
        For example 'geosplits/battle/full_data_set'
    """
    import os
    
    for f in [i for i in os.listdir(path) if not 'quick' in i]:
        # If quickload file doesn't already exist.
        if not os.path.isfile('%s/%s.quick'%(path,f)):
            indata=pickle.load(open('%s/%s'%(path,f),'rb'))

            del indata['geoSplit']

            pickle.dump(indata,open('%s/%s.quick'%(path,f),'wb'),-1)

