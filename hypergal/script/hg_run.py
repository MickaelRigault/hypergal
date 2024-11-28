#!/usr/bin/env python
# -*- coding: utf-8 -*-


import hypergal

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('filename')           # positional argument
    
    parser.add_argument('-z',"--redshift",  default=None, type=float,
                        help="target's redshit")
    
    parser.add_argument("--radec", default=None, type=str,
                        help="target's coordinates (deg,deg)")

    # ===
    parser.add_argument("--ncores", default=1, type=int,
                        help="number of cores used for sedfitting")

    parser.add_argument("--sedfitting", default=False, type=bool,
                        help="run sedfitting only.")
    
    args = parser.parse_args()
    # ============== #
    if args.redshift is None:
        raise ValueError("redshift requested")
    else:
        redshift = float(args.redshift)
        
    if args.radec is None:
        raise ValueError("radec requested")
    else:
        ra, dec = args.radec.split(",")
        radec = [float(ra), float(dec)]

        
    # ------------------- #
    #   Input Properties  #
    # ------------------- #
    run_properties = dict(cubefile=args.filename,
                          redshift=redshift, radec=radec,
                          ncores=args.ncores,
                          dasked=False)

        
    # ------------------- #
    #   SED Fitting only  #
    # ------------------- #
    if args.sedfitting:
        hypergal.run_sedfitting( **run_properties )
        print(f"sedfitting processing {args.filename} is done")
    # ------------------- #
    #   Full Hypergal     #
    # ------------------- #        
    else:
        hypergal.run_hypergal( **run_properties )
        print(f"full hypergal processing {args.filename} is done")
                
