
import numpy as np
from astropy.wcs import WCS as astropyWCS


class WCSHolder( object ):

    # =============== #
    #  Methods        #
    # =============== #
    # --------- #
    #  LOADER   #
    # --------- #        
    def load_wcs(self, header, pointingkey=["RA","DEC"]):
        """ """
        # - wcs
        self.set_wcs( astropyWCS(header), pointingkey=pointingkey)
        
        # - pointing        
        pra, pdec = header.get(pointingkey[0], None), header.get(pointingkey[1], None)
        if pra is None or pdec is None:
            return None
            
        from astropy import coordinates, units
        sc = coordinates.SkyCoord(pra, pdec, unit=(units.hourangle, units.deg))

        self.set_pointing(sc.ra.value, sc.dec.value)
        
    # --------- #
    #  SETTER   #
    # --------- #
    def set_wcs(self, wcs, pointingkey=["RA","DEC"]):
        """ """
        self._wcs = wcs

    def set_pointing(self, ra, dec):
        """ """
        self._pointing = ra, dec

    # --------- #
    #  GETTER   #
    # --------- #
    # --------- #
    #  Convert  #
    # --------- #

    # xy
    def xy_to_radec(self, x, y, reorder=True):
        """ get sky ra, dec [in deg] coordinates given the (x,y) quadrant positions  """
        if reorder and hasattr(self, "shape"):
            x = self.shape[1] -x -1 # starts at 0
            y = self.shape[0] -y -1 # starts at 0
                        
        return self.wcs.all_pix2world(np.asarray([np.atleast_1d(x),
                                                  np.atleast_1d(y)]).T,
                                      0).T

    # radec
    def radec_to_xy(self, ra, dec, reorder=True):
        """ get the (x,y) quadrant positions given the sky ra, dec [in deg] coordinates """
        
        x, y = self.wcs.all_world2pix(np.asarray([np.atleast_1d(ra),
                                                  np.atleast_1d(dec)]).T,
                                      0).T
        if reorder and hasattr(self, "shape"):
            x = self.shape[1] -x -1 # starts at 0
            y = self.shape[0] -y -1 # starts at 0

        return np.stack([x, y])

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def wcs(self):
        """ """
        if not hasattr(self, "_wcs"):
            return None
        return self._wcs

    def has_wcs(self):
        """ """
        return self.wcs is None
        
    @property
    def pointing(self):
        """ requested telescope pointing [in degree] """
        if not hasattr(self, "_pointing"):
            return None
            
        return self._pointing



def get_source_ellipses(sourcedf, sourcescale=5, system="xy",
                        wcs=None, wcsout=None, **kwargs):
    """ 
    Parameters
    ----------
    sourcedf: pandas.DataFrame
        This dataframe must contain the sep/sextractor ellipse information: \n
        x,y for the centroid\n
        a,b for the second moment (major and minor axis) \n
        theta for the angle (in degree) \n
        Must be in units of xy 
        
    sourcescale: float -optional-
        This multiply a and b. 1 means second moment (1 sigma)
           
    system: string -optional-
        Coordinate system of the returned ellipses \n
        - xy: native coordinates of the input source dataframe \n
        - radec: RA, Dec assuming the input wcs and the xy system.(xy->radec) \n
        - out: xy system of an alternative wcs solution (wcsout).(xy->radec->out) 
        
    wcs,wcsout: astropy WCS -optional-
        astropy WCS solution instance to convert xy<->radec \n
        wcs is needed if system is 'radec' or 'out' \n
        wcsout is need if system is 'out'

    Returns
    -------
    matplotlib patch (Ellipse/Polygon)
    
    """
    from matplotlib.patches import Ellipse, Polygon
    
    # = Baseline xy
    
    if system in ["xy", "pixels", "pix","pixel"]:
        return [ Ellipse( xy=(d.x,d.y),
                          width=d.a*sourcescale,
                          height=d.b*sourcescale,
                          angle=d.theta*180/np.pi)
                for d in sourcedf.itertuples()]
    
    # RA, Dec
    if system in ["radec", "world"]:
        if wcs is None:
            raise ValueError("no wcs provided. Necessary for radec projection")
        
        e_xy = get_source_ellipses(sourcedf, sourcescale=sourcescale, system="xy")
        return [Polygon(wcs.all_pix2world(e_.get_verts(), 0), **kwargs) for e_ in e_xy]
    
    # alternative xy    
    elif system in ["out", "xyout", "xy_out"]:
        if wcsout is None or wcs is None:
            raise ValueError("no wcs or no wcsout provided. Necessary for out projection")
        e_radec = get_source_ellipses(sourcedf, sourcescale, system="radec", wcs=wcs)
        return [Polygon(wcsout.all_world2pix(e_.get_verts(), 0), **kwargs) for e_ in e_radec]
        
    else:
        raise NotImplementedError(f"Only xy, radec or out system implemented, {system} given")
