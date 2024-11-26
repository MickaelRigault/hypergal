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
    parser.add_argument("--ncores", default=5, type=int,
                        help="number of cores used for sedfitting")

    
    args = parser.parse_args()
    # ============== #
    if args.redshift is None:
        raise ValueError("redshift requested")
        
    if args.radec is None:
        raise ValueError("radec requested")
    else:
        ra, dec = args.radec.split(",")
        radec = [float(ra), float(dec)]

    
    hypergal.run_sedfitting(cubefile=args.filename,
                            redshift=redshift, radec=radec,
                            ncores=args.ncores,
                            dasked=False)
                            
    print(f"processing {args['filename']} is Done")
                
