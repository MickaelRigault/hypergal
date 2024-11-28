""" top level script """
import os
import numpy as np


def run_hypergal(cubefile, radec, redshift, spxy = None,
                 dasked=True,
                 filters=["ps1.g", "ps1.r", "ps1.i", "ps1.z", "ps1.y"],
                 filters_fit=["ps1.g", "ps1.r", "ps1.i", "ps1.z"],
                 size=180, binfactor=2,
                 source_filter="ps1.r", source_thres=2, limit_pos=None,
                 #
                 psfmodel="Gauss2D", pointsourcemodel="GaussMoffat2D",
                 #
                 sn_only=False, host_only=False, rmtarget=None,
                 scale_cout=15, scale_sedm=10, target_radius=10,
                 use_exist_intcube=True, ncores=None,
                 # slices
                 lbda_range=[5000, 8500], nslices=6,
                 curved_bkgd=True,
                ):
    """ top level function to run hypergal

    Parameters
    ----------
    cubefile: str, Path
        location of the cube to process.

    radec: (float, float)
        coordinate of the targets in degree (RA, Dec)

    redshift: float
        redshift of the target (used by SED fitting)

    spxy: (flaot, float)
        coordinates of the target in the cube (used as initial guess)
        If None, this is guess from derived radec.
    
    # Dasking

    dasked: bool
        should this use dask to run the pipeline (delayed and persists)

    # Cutouts
    
    filters: list
        list of filters to be used by hypergal SEDfitting
        (warnings: hypergal current only works for PS1 filters)
        
    filters_fit: list
        list of filters used to fit individual slices 
        used to derive ADR- and PSF-model.

    size: int
        size of the photometric image cutouts
        
    binfactor: int
        rebinning factor to reduce the size of the cutouts
        
    source_filter: str
        name of the filter used to extract image sources

    source_thres: float
        threshold for the source extraction (see sep.extract()

    # hypergal model

    sn_only: bool

    host_only: bool
    
    rmtarget: None

    scale_cout: float

    scale_sedm: float

    target_radius: float

    ncores: int, None
    
    lbda_range: (float, float)
        wavelength range used to fit meta-slices to constrain ADR and PSF model
        (see nslices)

    nslices: int
        number of slices used to fit meta-slices to constrain ADR and PSF model
        (see lbda_range)

    Returns
    -------
    hg_spec: spectrum
        the resulting transient model 

    storing: list
        list of stored elements.

    """
    # pysedm
    from pysedm.sedm import SEDM_LBDA
    
    # internal imports
    from . import io
    from . import psf
    from .spectroscopy.basics import get_calibrated_cube
    from .photometry.panstarrs import get_cutout
    from .photometry.basics import get_filter    
    from .scene.basics import SEDM_to_PS1_SCALERATIO, PointSource
    from .fit import SceneFitter, MultiSliceParameters, Priors, slicefit_to_target_spec, build_hg_cubes, get_sourcedf, fit_cube
    from .spectroscopy.sedfitting import run_sedfitter
    if dasked:
        import dask
        get_calibrated_cube = dask.delayed(get_calibrated_cube)
        get_cutout = dask.delayed(get_cutout)
        MultiSliceParameters = dask.delayed(MultiSliceParameters)
    
    # Parsing on working directory
    info = io.parse_filename(cubefile)
    filedir = os.path.dirname(cubefile)
    working_dir = os.path.join(filedir, f"tmp_{info['sedmid']}")
    
    # ------------------- #
    #  Step 1: Loading    #
    # ------------------- #
    
    # 1.1 load the cube
    cube = get_calibrated_cube(cubefile, radec=radec, spxy=spxy)    
    if dasked:
        cube = cube.persist() # make sense to persist as often used.
    
    # 1.2 fetch corresponding cutouts
    cutouts = get_cutout(radec=radec, filters=filters, size=size)

    # and rebin it to lower the number of pixel to model
    cout_cube = cutouts.to_cube(binfactor=binfactor)
    if dasked:
        cout_cube = cout_cube.persist() # used a lot.

    # 1.3 extract sources from the refere cutouts    
    sources = cutouts.extract_sources(filter_=source_filter, thres=source_thres, savefile=None)
    if dasked:
        sources = sources.persist()

    
    # 1.4 get tuned SEDm- and cutout-cubes
    source_coutcube = cout_cube.get_extsource_cube(sourcedf=sources, 
                                               wcsin=cout_cube.wcs, 
                                               radec=radec,
                                               sourcescale=scale_cout, radius=target_radius*SEDM_to_PS1_SCALERATIO, 
                                               boundingrect=True, sn_only=sn_only)

    source_sedmcube = cube.get_extsource_cube(sourcedf=sources, wcsin=cout_cube.wcs, 
                                               radec=radec,
                                                sourcescale=scale_sedm, radius=target_radius, 
                                               boundingrect=False, sn_only=sn_only)
    
    if rmtarget is not None:
        rmradius = 2
        target_pos = source_sedmcube.radec_to_xy(*radec).flatten()
        source_sedmcube = source_sedmcube.get_target_removed(target_pos=target_pos,radius=rmradius)

    
    # --------------------- #
    #  Step 2: Fit CutOuts  #
    # --------------------- #

    ## 2.1: get meta-slices
    cout_filter_slices = {f_: source_coutcube.get_slice(index=filters.index(f_), slice_object=True)
                              for f_ in filters_fit}

    source_sedmcube_sub = source_sedmcube.get_partial_cube(source_sedmcube.indexes,
                                                           # limit to where SEDm is well defined.
                                                           slice_id = np.argwhere((SEDM_LBDA > 4500) & (SEDM_LBDA < 8800)).squeeze()
                                                           )
    sedm_filter_slices = {f_: source_sedmcube_sub.get_slice(lbda_max=np.max(get_filter(f_, as_dataframe=False)[0]),
                                                        lbda_min=np.min(get_filter(f_, as_dataframe=False)[0]),
                                                        slice_object=True) for f_ in filters_fit}

    ## 2.2 get coordinate information
    xy_in = source_coutcube.radec_to_xy(*radec).flatten()
    xy_comp = source_sedmcube_sub.radec_to_xy(*radec).flatten()
    mpoly = source_sedmcube_sub.get_spaxel_polygon(format='multipolygon')

    ## 2.3 fit meta-slices
    _fit_slices_projection = SceneFitter.fit_slices_projection
    if dasked:
        _fit_slices_projection = dask.delayed(_fit_slices_projection)
        
    best_fits = {}
    for f_ in filters_fit:
        gm = psf.gaussmoffat.GaussMoffat2D(alpha=2.5, eta=1)
        if dasked:
            ps = dask.delayed(PointSource)(gm, mpoly)
        else:
            ps = PointSource(gm, mpoly)
            
        psfmodel_ = getattr(psf, psfmodel)()
        
        fitted_slices = _fit_slices_projection(cout_filter_slices[f_],
                                                  sedm_filter_slices[f_],
                                                  psf=psfmodel_,
                                                  savefile=None,
                                                  whichscene='SceneSlice',
                                                  pointsource=ps,
                                                  xy_in=xy_in,
                                                  xy_comp=xy_comp,
                                                  guess=None, add_lbda=True, 
                                                  priors=Priors(),
                                                  host_only=host_only,
                                                  kind='metaslices', onlyvalid=True)
        if dasked:
            fitted_slices = fitted_slices.persist()
            
        best_fits[f_] = fitted_slices

    ## 2.4 build the corresponding MultiSlice object        
    cout_ms_param = MultiSliceParameters(best_fits, cubefile=cubefile,
                                          psfmodel=psfmodel.replace("2D", "3D"),
                                          pointsourcemodel=pointsourcemodel.replace("2D", "3D"),
                                          load_adr=True, load_psf=True,
                                          load_pointsource=True)

    # ----------------- #
    #  Step 3: SED      #
    # ----------------- #

    intcube_filepath = io.e3dfilename_to_hgcubes(cubefile, "intcube")
    if use_exist_intcube and os.path.exists(intcube_filepath):
        from .spectroscopy import WCSCube
        if dasked:
            read_wcscube = dask.delayed(WCSCube.read_hdf)
        else:
            read_wcscube = WCSCube.read_hdf
            
        int_cube = read_wcscube(intcube_filepath)            

    else:
        saveplot_rmspull = None# plotbase + '_' + name + '_cigale_pullrms.png'
        saveplot_intcube = None# plotbase + '_' + name + '_intcube.png'
        if dasked:
            run_sedfitter = dask.delayed(run_sedfitter)
                
        int_cube = run_sedfitter(source_coutcube,
                                         redshift=redshift, working_dir=working_dir,
                                         sedfitter="cigale", lbda=SEDM_LBDA,
                                         saveplot_rmspull=saveplot_rmspull,
                                         saveplot_intcube=saveplot_intcube, 
                                         sn_only=sn_only, ncores=ncores,
                                         fileout=intcube_filepath # this store
                                )
    # persist as key element.
    if dasked:
        int_cube = int_cube.persist()
    
    # ------------------- #
    #  Step 4: ADR & PSF  #
    # ------------------- #
    ## 4.1 define meta-slices
    mcube_sedm = source_sedmcube.to_metacube(lbda_range, nslices=nslices)
    mcube_intr = int_cube.to_metacube(lbda_range, nslices=nslices)

    ## 4.2 run meta slices
    saveplot_structure = None # plotbase + '_' + name + '_metaslice_fit_'
    bestfit_mfit = fit_cube(mcube_sedm, mcube_intr, radec,
                                #
                                dasked=dasked, # use dask or not ?
                                #
                                nslices=nslices,
                                saveplot_structure=saveplot_structure,
                                mslice_param=cout_ms_param, 
                                psfmodel=psfmodel, 
                                pointsourcemodel=pointsourcemodel,  
                                curved_bkgd=curved_bkgd, 
                                limit_pos=limit_pos,
                                fix_params=['scale', 'rotation'], 
                                host_only=host_only, sn_only=sn_only, 
                                kind='metaslices', onlyvalid=True)
    
    ## 4.3 get parameter guesser
    saveplot_adr = None # plotbase + '_' + name + '_adr_fit.png'
    saveplot_psf = None # plotbase + '_' + name + '_psf3d_fit.png'
    
    # MultiSliceParameters is already delayed or not
    meta_ms_param = MultiSliceParameters( bestfit_mfit, cubefile=cubefile,
                                             psfmodel=psfmodel.replace("2D", "3D"), 
                                             pointsourcemodel=pointsourcemodel.replace("2D", "3D"),
                                             load_adr=True, load_psf=True, 
                                             load_pointsource=True, 
                                             saveplot_adr=saveplot_adr, 
                                             saveplot_pointsource=saveplot_psf)

    # ------------------- #
    #  Step 5: Amplitude  #
    # ------------------- #
    ## 5.1 fit cube assuming ADR- and PSF-model (amplitude free)
    bestfit_completfit = fit_cube( source_sedmcube, int_cube, radec,
                                      dasked=dasked,
                                      nslices=len(SEDM_LBDA),
                                      mslice_param=meta_ms_param, 
                                      psfmodel=psfmodel, pointsourcemodel=pointsourcemodel, 
                                      #jointfit=False, 
                                      curved_bkgd=curved_bkgd,
                                      saveplot_structure=None,  # plotbase+"full_fit_",
                                      fix_params=['scale', 'rotation', "xoff", "yoff",
                                            "a", "b", "sigma", 'a_ps', 'b_ps', 'alpha_ps', 'eta_ps'],
                                      host_only=host_only, sn_only=sn_only, 
                                      kind='slices')

    if dasked:
        bestfit_completfit = bestfit_completfit.persist()
        
    ## 5.2 and get the MultiSliceParameter
    full_ms_param = MultiSliceParameters(bestfit_completfit, 
                                                   psfmodel=psfmodel.replace("2D", "3D"), 
                                                   pointsourcemodel=pointsourcemodel.replace("2D", "3D"),
                                                   load_adr=False, load_psf=False, 
                                                   load_pointsource=False)
    if dasked:
        full_ms_param = full_ms_param.persist()

    # ------------------- #
    #  Step 6: Storing    #
    # ------------------- #
    storing = []
    
    ## 6.1: final hypergal SN spectrum
    if dasked:
        slicefit_to_target_spec = dask.delayed(slicefit_to_target_spec)
        
    hg_spec = slicefit_to_target_spec(bestfit_completfit, cube.header)
    # then store
    fileout_hgspec = io.e3dfilename_to_hgspec(cubefile, 'target')
    storing.append( hg_spec.writeto(fileout_hgspec, ascii=True) )

    ## 6.2: individual fit components
    hostmodel, snmodel, bkgdmodel = build_hg_cubes(int_cube, cube, radec,
                                                       dasked=dasked,
                                                       mslice_meta=meta_ms_param, mslice_final=full_ms_param,
                                                       psfmodel=psfmodel, pointsourcemodel=pointsourcemodel, 
                                                       curved_bkgd=curved_bkgd)
    # storing host model
    storing.append( hostmodel.to_hdf( io.e3dfilename_to_hgcubes(cubefile, "hostmodel")) )
    # storing snmodel model
    storing.append( snmodel.to_hdf( io.e3dfilename_to_hgcubes(cubefile, "snmodel")) )
    # storing background model
    storing.append( bkgdmodel.to_hdf( io.e3dfilename_to_hgcubes(cubefile, "bkgdmodel")) )

    ## 6.3: summary plot
    from .utils.plot import global_report
    if dasked:
        global_report = dask.delayed(global_report)

        
    fig = global_report(cube, hostmodel, snmodel, bkgdmodel, source_coutcube, 
                        get_sourcedf(radec, cubefile, size), 
                        bestfit_completfit, bestfit_mfit, radec, redshift, cubefile, lbda_range, nslices, 
                        saveplot=None)
    
    figfileout = fileout_hgspec.replace("hgspec", "hgreport").replace(".txt",".pdf")
    storing.append( fig.savefig(figfileout) )
    
    # output
    return hg_spec, storing


# ============= #
#  Only SED     #
# ============= #
def run_sedfitting(cubefile, radec, redshift, spxy=None,
                       dasked=True,
                       filters=["ps1.g", "ps1.r", "ps1.i", "ps1.z", "ps1.y"],
                       size=180, binfactor=2,
                       source_filter="ps1.r", source_thres=2,
                       #
                       sn_only=False, scale_cout=15, target_radius=10,
                       ncores=None):
    """ 
    """
    # pysedm
    from pysedm.sedm import SEDM_LBDA
    
    # internal imports
    from . import io
    from .spectroscopy.basics import get_calibrated_cube
    from .spectroscopy.sedfitting import run_sedfitter    
    from .photometry.basics import get_filter
    from .photometry.panstarrs import get_cutout
    from .scene.basics import SEDM_to_PS1_SCALERATIO


    if dasked:
        import dask
        get_calibrated_cube = dask.delayed(get_calibrated_cube)
        get_cutout = dask.delayed(get_cutout)

    
    # Parsing on working directory
    info = io.parse_filename(cubefile)
    filedir = os.path.dirname(cubefile)
    working_dir = os.path.join(filedir, f"tmp_{info['sedmid']}")
    
    # ------------------- #
    #  Step 1: Loading    #
    # ------------------- #
    
    # 1.1 load the cube
    cube = get_calibrated_cube(cubefile, radec=radec, spxy=spxy)
    if dasked:
        cube = cube.persist() # make sense to persist as often used.
    
    # 1.2 fetch corresponding cutouts
    cutouts = get_cutout(radec=radec, filters=filters, size=size)

    # and rebin it to lower the number of pixel to model
    cout_cube = cutouts.to_cube(binfactor=binfactor)
    if dasked:
        cout_cube = cout_cube.persist() # used a lot.

    # 1.3 extract sources from the refere cutouts    
    sources = cutouts.extract_sources(filter_=source_filter, thres=source_thres, savefile=None)
    if dasked:
        sources = sources.persist()

    # 1.4 get tuned SEDm- and cutout-cubes
    source_coutcube = cout_cube.get_extsource_cube(sourcedf=sources,
                                                       wcsin=cout_cube.wcs, 
                                                       radec=radec,
                                                       sourcescale=scale_cout,
                                                       radius=target_radius*SEDM_to_PS1_SCALERATIO, 
                                                       boundingrect=True, sn_only=sn_only)

  
    # ----------------- #
    #  Step 2: SED      #
    # ----------------- #

    intcube_filepath = io.e3dfilename_to_hgcubes(cubefile, "intcube")
    
    saveplot_rmspull = None# plotbase + '_' + name + '_cigale_pullrms.png'
    saveplot_intcube = None# plotbase + '_' + name + '_intcube.png'
    
    if dasked:
        run_sedfitter = dask.delayed(run_sedfitter)

    int_cube = run_sedfitter(source_coutcube,
                                 redshift=redshift, working_dir=working_dir,
                                 sedfitter="cigale", lbda=SEDM_LBDA,
                                 saveplot_rmspull=saveplot_rmspull,
                                 saveplot_intcube=saveplot_intcube, 
                                 sn_only=sn_only, ncores=ncores,
                                 fileout=intcube_filepath # this store
                            )
    # persist as key element.
    if dasked:
        int_cube = int_cube.persist()

    return int_cube
