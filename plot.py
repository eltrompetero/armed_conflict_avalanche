import numpy as np


def plot_pix_boundaries(m,Nside,n_phi=1000,plot_kw={'c':'k'}):
    """
    Plot the boundaries of the pixels given by healpy on to a given geographic projection.
    
    Parameters
    ----------
    m : matplotlib.Basemap
    Nside : int
    n_phi : int,1000
    plot_kw : dict,{'c':'k'}
    
    Returns
    -------
    """
    from numpy import pi,nan

    phi=np.linspace(1/n_phi,2*pi-1/n_phi,n_phi)
    phit=phi%(pi/2)

    # Polar caps
    for k in range(Nside):
        z1=1-k**2/(3*Nside**2) * (pi/2/phit)**2
        z2=1-k**2/(3*Nside**2) * (pi/(2*phit-pi))**2
        
        z1[z1<=(2/3)]=nan
        z2[z2<=(2/3)]=nan
        
        z1,ix1=split_by_nan(z1)
        z2,ix2=split_by_nan(z2)
        
        for z1_,ix1_,z2_,ix2_ in zip(z1,ix1,z2,ix2):
            m.plot((phi[ix1_]/pi*180-180), z1_*90,latlon=True,**plot_kw)
            m.plot((phi[ix1_]/pi*180-180),-z1_*90,latlon=True,**plot_kw)
            m.plot((phi[ix2_]/pi*180-180), z2_*90,latlon=True,**plot_kw)
            m.plot((phi[ix2_]/pi*180-180),-z2_*90,latlon=True,**plot_kw)

    # Base pixels
    for k in range(4):
        m.plot([k/2*180-180]*2,[ 2/3*90, 1*90],latlon=True)
        m.plot([k/2*180-180]*2,[-2/3*90,-1*90],latlon=True)

    # Equatorial region
    for k in range(6*Nside):  # unsure of what max value of k should be
        z1=2/3-4*k/(3*Nside)+8*phi/3/pi
        z2=z1-16*phi/3/pi

        z1[np.abs(z1)>(2/3)]=nan
        z2[np.abs(z2)>(2/3)]=nan
        
        z1,ix1=split_by_nan(z1)
        z2,ix2=split_by_nan(z2)
        
        for z1_,ix1_ in zip(z1,ix1):
            m.plot((phi[ix1_]/pi*180-180), z1_*90,latlon=True,**plot_kw)
            m.plot((phi[ix1_]/pi*180-180),-z1_*90,latlon=True,**plot_kw)
        for z2_,ix2_ in zip(z2,ix2):
            m.plot((phi[ix2_]/pi*180-180), z2_*90,latlon=True,**plot_kw)
            m.plot((phi[ix2_]/pi*180-180),-z2_*90,latlon=True,**plot_kw)
