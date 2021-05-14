""" Module Containing the basic objects """
import warnings
import numpy as np

from ..photometry.astrometry import WCSHolder
from pyifu.spectroscopy import Cube

class WCSCube( Cube, WCSHolder ):
    """ """

    @classmethod
    def read_sedmfile(cls, cubefile):
        """ """
        from pysedm import get_sedmcube
        return cls.from_sedmcube( get_sedmcube(cubefile) )
        
    @classmethod
    def from_sedmcube(cls, cube):
        """ """
        from pysedm import astrometry
        from astropy.io import fits
        
        wcsdict = astrometry.get_wcs_dict(cube.filename)
        
        keys = ["EXPTIME", "ADCSPEED","TEMP","GAIN_SET", "ADC","MODEL","SNSR_NM","SER_NO","TELESCOP",
                "GAIN","CAM_NAME","INSTRUME","UTC","END_SHUT","OBSDATE","OBSTIME","LST","MJD_OBS",
                "JD","APPEQX","EQUINOX","RA","TEL_RA","DEC","TEL_DEC","TEL_AZ","TEL_EL","AIRMASS",
                "TEL_PA","RA_OFF","DEC_OFF","TELHASP","TELDECSP","FOCPOS","IFUFOCUS","IFUFOC2",
                "DOMEST","DOMEMO","DOME_GAP","DOMEAZ","OBJECT","OBJTYPE","IMGTYPE","OBJNAME",
                "OBJEQX","OBJRA","OBJDEC","ORA_RAT","ODEC_RAT","SUNRISE","SUNSET","TEL_MO",
                "SOL_RA","SOL_DEC","WIND_DIR","WSP_CUR","WSP_AVG","OUT_AIR","OUT_HUM","OUT_DEW",
                "IN_AIR","IN_HUM","IN_DEW","MIR_TEMP","TOP_AIR","WETNESS","FILTER","NAME",
                "P60PRID","P60PRNM","P60PRPI","REQ_ID","OBJ_ID",
                "ENDAIR","ENDDOME","END_RA","END_DEC","END_PA","BIASSUB","BIASSUB2",
                "CCDBKGD","ORIGIN","FLAT3D","FLATSRC","ATMCORR","ATMSRC","ATMSCALE","IFLXCORR",
                "IFLXREF","CCDIFLX"]
        nheader = {k:cube.header[k] for k in keys}
        cube.set_header(fits.Header({**nheader,**wcsdict}))
        
        return cls.from_data(data=cube.data, 
                             variance=cube.variance, 
                             lbda=cube.lbda,header=cube.header,
                             spaxel_vertices=cube.spaxel_vertices,
                                spaxel_mapping=cube.spaxel_mapping)
    
        
        
    @classmethod
    def from_cutouts(cls, hgcutout, header_id=0, influx=True, binfactor=None, xy_center=None):
        """ """
        
        lbda = np.asarray(hgcutout.lbda)
        sort_lbda = np.argsort(lbda)

        # Data
        lbda = lbda[sort_lbda]
        data = hgcutout.get_data(influx=influx)[sort_lbda]
        variance = hgcutout.get_variance(influx=influx)[sort_lbda]
        spaxel_vertices = np.asarray([[0,0],[1,0],[1,1],[0,1]])-0.5 # centered
        if binfactor is not None:
            binfactor = int(binfactor)
            if binfactor==1:
                warnings.warn("binfactor=1, this means nothing to do.")
            else:
                from ..utils.array import restride
                data = np.sum(restride(data, (1, binfactor, binfactor)),axis=(-2,-1))
                variance = np.sum(restride(variance, (1, binfactor, binfactor)),axis=(-2,-1))
                spaxel_vertices *=binfactor
        else:
            binfactor=1
            
        # Header
        header = hgcutout.instdata[header_id].header

        # Mapping
        xsize, ysize = np.asarray(data[header_id].shape)
        pixels_ = np.mgrid[0:xsize*binfactor:binfactor,0:ysize*binfactor:binfactor]
        
        init_shape = np.shape(pixels_)
        spaxels_xy = pixels_.reshape(2, init_shape[1]*init_shape[2]).T
        if xy_center is not None:
            spaxels_xy = np.asarray(spaxels_xy, dtype="float")-xy_center
        spaxel_mapping = {i:v for i,v in enumerate(spaxels_xy)}

        #
        # Init
        return cls.from_data(data=np.concatenate(data.T, axis=0).T, 
                              variance=np.concatenate(variance.T, axis=0).T, 
                            lbda=lbda,header=header,
                            spaxel_vertices=spaxel_vertices,
                                spaxel_mapping=spaxel_mapping)
    
    # ================ #
    #   Methods        #
    # ================ #
    def set_header(self, header, *args, **kwargs):
        """ """
        _ = super().set_header(header)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.load_wcs(header)
        
    def get_target_removed(self, target_pos=None, radious=3, store=False, **kwargs):
        """ """
        from . import sedmtools
        if target_pos is None:
            target_pos = sedmtools.get_target_position(self)
            
        return sedmtools.remove_target_spx(self, target_pos, radius=radious, store=store, **kwargs)



    def get_extsource_cube(self, sourcedf, wcsin, wcsout=None, sourcescale=5, 
                          boundingrect=False, slice_id=None):
        """ """
        from shapely.geometry import Polygon
        if wcsout is None:
            wcsout = self.wcs

        e_out = astrometry.get_source_ellipses(sourcedf, wcs=wcsin,  wcsout=wcsout, system="out", 
                                               sourcescale=sourcescale)
        if boundingrect:
            [xmin, ymin], [xmax, ymax] = np.percentile(np.concatenate([e_.xy for e_ in e_out]), [0,100], axis=0)
            polys = [Polygon([[xmin, ymin],[xmin, ymax],[xmax, ymax], [xmax, ymin]])]
        else:
            polys = [Polygon(e_.xy) for e_ in e_out]

        spaxels = np.unique(np.concatenate([self.get_spaxels_within_polygon(poly_) for poly_ in polys]))
        if slice_id is None:
            slice_id  = np.arange( len(self.lbda) )

        return self.get_partial_cube(spaxels,  slice_id)
