__version__ = "2.0.0"


import os
_PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))

from .hypergal import run_hypergal, run_sedfitting
