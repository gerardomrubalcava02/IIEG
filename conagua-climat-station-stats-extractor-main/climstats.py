# ==============================================================================
# Project Name: Extracting Climatological Statistical Information Application
# Script Name: climstats.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   The application allows users to consult and extract historical information 
#   from the conventional weather stations that make up the National Network 
#   of CONAGUA.
#
# License: MIT
# ==============================================================================

import os

from dataclasses import dataclass
from jsonargparse import CLI
from core.tools import (
    extract_climate_station_data,
    remove_files_by_extension,
    generate_geopackage,
    Stations
)


@dataclass
class ClimateStats:
    """
    Application that handles climate statistics extraction from specified stations.
    
    Climate stations come from [CONAGUA](https://smn.conagua.gob.mx/es/climatologia/informacion-climatologica/informacion-estadistica-climatologica)
    """

    def extract(
            self,
            stations: Stations,
            url: str,
            output_path: os.PathLike,
            max_workers: int | None = None,
            request_delay_seconds: float = 0.0,
            timeout_seconds: float = 30.0,
            max_retries: int = 3,
            backoff_factor: float = 1.0
    ) -> None:
        """Extract climate station data and save it to the specified output path.

        Args:
            stations (Stations): An instance of Stations containing the relevant climate stations.
            url (str): The URL to retrieve climate data.
            output_path (os.PathLike): The file path where the extracted data will be stored.
        """
        extract_climate_station_data(
            stations=stations,
            url=url,
            output_path=output_path,
            max_workers=max_workers,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )
        generate_geopackage(output_path=output_path)
        remove_files_by_extension(directory=output_path)


if __name__ == '__main__':
    CLI(ClimateStats)
