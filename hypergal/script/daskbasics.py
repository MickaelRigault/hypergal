import os
import warnings
import numpy as np
import pandas

from pysedm.dask import base
from dask import delayed

from ..photometry import basics as photobasics
from ..photometry import panstarrs
from ..spectroscopy import basics as spectrobasics
from ..spectroscopy import sedfitting
from ..fit import SceneFitter
from .. import psf



class DaskHyperGal( base.DaskCube ):

    
    @classmethod
    def get_sourcecubes(cls, cubefile, radec, binfactor=2,
                            filters=["ps1.g","ps1.r", "ps1.i","ps1.z","ps1.y"],
                            source_filter="ps1.r", source_thres=2,
                            scale_cout=15, scale_sedm=10, rmtarget=2):
        """ """
        #
        # Cubes
        sedm_cube = cls.get_calibrated_cube(cubefile, as_wcscube=True, apply_byecr=True)
        cutouts   = cls.get_cutout(radec=radec, binfactor=2, filters=filters)
        #
        # cout_cube->Source & cube
        sources   = cutouts.extract_sources(filter_=source_filter, thres=source_thres,
                                               savefile=None)
        cout_cube = cutouts.to_cube(binfactor=binfactor)
        #
        # get sources cube
        wcsin = cout_cube.wcs
        source_coutcube = cout_cube.get_extsource_cube(sourcedf=sources, wcsin=wcsin, 
                                                 sourcescale=scale_cout, boundingrect=True)

        source_sedmcube = sedm_cube.get_extsource_cube(sourcedf=sources, wcsin=wcsin, 
                                                 sourcescale=scale_sedm, boundingrect=False)
        if rmtarget is not None:
            rmradius = 2
            target_pos = source_sedmcube.radec_to_xy(*radec).flatten()

            source_sedmcube_notarget = source_sedmcube.get_target_removed(target_pos=target_pos, 
                                                                 radius=rmradius, 
                                                                 store=False, get_filename=False)
            return source_coutcube,source_sedmcube_notarget
        
        return source_coutcube,source_sedmcube

    @staticmethod
    def fit_cout_slices(source_coutcube, source_sedmcube, radec,
                          filterin=["ps1.g","ps1.r", "ps1.i","ps1.z","ps1.y"],
                          filters_to_use=["ps1.r", "ps1.i","ps1.z"],
                          psfmodel="Gauss2D"):
        """ """
        #
        # Get the slices
        cout_filter_slices = {f_: source_coutcube.get_slice(index=filterin.index(f_), slice_object=True) 
                                  for f_ in filters_to_use}

        sedm_filter_slices = {f_: source_sedmcube.get_slice(lbda_trans=photobasics.get_filter(f_, as_dataframe=False), 
                                                            slice_object=True)
                                  for f_ in filters_to_use}
        xy_in   = source_coutcube.radec_to_xy(*radec).flatten()
        xy_comp = source_sedmcube.radec_to_xy(*radec).flatten()
        #
        # Get the slices
        best_fits = {}
        for f_ in filters_to_use:
            savefile = None
            best_fits[f_] = delayed(SceneFitter.fit_slices_projection)(cout_filter_slices[f_], 
                                                                            sedm_filter_slices[f_], 
                                                                            psf=getattr(psf,psfmodel)(), 
                                                                            savefile=savefile, 
                                                                            xy_in=xy_in, 
                                                                            xy_comp=xy_comp)
        return delayed(pandas.concat)(best_fits)

    # =============== #
    #   INTERNAL      #
    # =============== #
    @classmethod
    def get_calibrated_cube(cls, cubefile, fluxcalfile=None, apply_byecr=True,
                            store_data=False, get_filename=False, as_wcscube=True, **kwargs):
        """ """
        cube = super().get_calibrated_cube(cubefile, fluxcalfile=fluxcalfile,
                                               apply_byecr=apply_byecr,
                                               get_filename=False, **kwargs)
        if not as_wcscube:
            return cube
        
        if get_filename and not store_data:
            warnings.warn("you requested get_filename without storing the data (store_data=False)")
            
        return delayed(spectrobasics.sedmcube_to_wcscube)(cube, 
                                                          store_data=store_data, 
                                                          get_filename=get_filename)
    @staticmethod
    def get_cutout(radec=None, cubefile=None, client_dl=None, filters=None, **kwargs):
        """ """
        prop_cutout = dict(filters=filters, client=client_dl)
        if cubefile is not None:
            return delayed(panstarrs.PS1CutOuts.from_sedmfile)(cubefile, **prop_cutout)
        if radec is not None:
            return delayed(panstarrs.PS1CutOuts.from_radec)(*radec, **prop_cutout)

        raise ValueError("cubefile or radec must be given. Both are None")

    @staticmethod
    def run_sedfitter(cube_cutout, redshift, working_dir, sedfitter="cigale", ncores=1, lbda=None):
        """ """
        if lbda is None:
            from pysedm.sedm import SEDM_LBDA
            lbda = SEDM_LBDA

        tmp_inputpath = os.path.join(working_dir, "input_sedfitting.txt")

        if sedfitter == "cigale":
            sfitter = delayed(sedfitting.Cigale.from_cube_cutouts)(cube_cutout, redshift, 
                                                                   tmp_inputpath=tmp_inputpath,
                                                                  initiate=True, 
                                                                  working_dir=working_dir,
                                                                  ncores=ncores)
        else:
            raise NotImplementedError(f"Only the cigale sed fitted has been implemented. {sedfitter} given")
        
        # run sedfit
        bestmodel_dir = sfitter.run() # bestmodel_dir trick is for dask

        # get the results
        spectra_lbda = sfitter.get_sample_spectra(bestmodel_dir=bestmodel_dir, 
                                                  lbda_sample=lbda)
        specdata = spectra_lbda[0]
        lbda = spectra_lbda[1]


        return cube_cutout.get_new(newdata=specdata, newlbda=lbda, newvariance="None")