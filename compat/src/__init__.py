"""Compatibility alias: ``from src import ...`` keeps working.

The package was renamed ``src`` -> ``tender_analysis`` when it was packaged
(it was previously imported by putting the repo root on ``sys.path``, which
only works from a checkout). Every existing notebook opens with
``from src import OnePot, OnePotRIXS, index_beamtime``, so that spelling is
re-exported here rather than broken. New code should import
``tender_analysis`` directly.
"""

from tender_analysis import *  # noqa: F401,F403
from tender_analysis import __all__  # noqa: F401
