# ==============================================================================
# Project Name: Extracting Climatological Statistical Information Application
# Script Name: patterns.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   The application allows users to consult and extract historical information 
#   from the conventional weather stations that make up the National Network 
#   of CONAGUA.
#
# License: MIT
# ==============================================================================

from dataclasses import dataclass


@dataclass
class Pattern:
    """Refex patterns to find the climate station fields"""

    @property
    def station_name(self):
        return r"NOMBRE\s+:\s+[a-zA-Z ()]+"

    @property
    def municipality_name(self):
        return r"MUNICIPIO\s+:\s+[a-zA-Z ()]+"

    @property
    def lat(self):
        return r"LATITUD\s+:\s+\-?\d+\.?\d+"

    @property
    def lon(self):
        return r"LONGITUD\s+:\s+\-?\d+\.?\d+"

    @property
    def height(self):
        return r"ALTITUD\s+:\s+\-?\d+"

    @property
    def station_status(self):
        return r"SITUACION\s+:\s+[a-zA-Z ()]+"

    @property
    def station_state(self):
        return r"[a-z]+"
    
    @property
    def station_number(self):
        return r"[0-9]+"
