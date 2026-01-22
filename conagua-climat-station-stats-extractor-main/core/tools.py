# ==============================================================================
# Project Name: Extracting Climatological Statistical Information Application
# Script Name: tools.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   The application allows users to consult and extract historical information 
#   from the conventional weather stations that make up the National Network 
#   of CONAGUA.
#
# License: MIT
# ==============================================================================

import glob
import multiprocessing
import os
import random
import time

import geopandas as gpd
import pandas as pd

from tqdm import tqdm
from shapely.geometry import Point
from utils.logger import Logger
from core.extractor import ClimateExtractor
from pandas import DataFrame

type Stations = list[str]


def process_station(
        args: tuple[int, str, str, float, float, int, float]
    ) -> None:
    """Process a single climate station to extract data.

    Args:
        args (tuple): A tuple containing a station, URL, and output path.
    """
    station, url, output_path, request_delay_seconds, timeout_seconds, max_retries, backoff_factor = args
    Logger.debug('🚀 Running Data Extraction for Climate Station: {}'.format(station))
    if request_delay_seconds > 0:
        time.sleep(random.uniform(0, request_delay_seconds))
    try:
        extractor = ClimateExtractor(
            station=station,
            url=url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )
        extractor.run()
        result = extractor.get()
        if result is not None:
            save_climte_station_data(
                station_stats=result,
                climate_station=station,
                output_path=output_path
            )
            Logger.debug(f'🟢 Station [{station}] processed.')
    except Exception as e:
        Logger.error(f"Error processing station {station}: {str(e)}")
    Logger.debug(f'🟢 Extraction for Station {station} Done.')


def extract_climate_station_data(
        stations: list,
        url: str,
        output_path: os.PathLike,
        max_workers: int | None = None,
        request_delay_seconds: float = 0.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ) -> None:
    """Extract climate data for the specified stations and save it to the output path.

    Args:
        stations (list): A list of climate station IDs.
        url (str): The URL to retrieve climate data.
        output_path (os.PathLike): The directory path where the extracted data will be stored.
        max_workers (int | None): Max number of worker processes to use (defaults to CPU count).
        request_delay_seconds (float): Max jittered delay before each request to reduce bursts.
        timeout_seconds (float): Per-request timeout for the HTTP call.
        max_retries (int): Number of retries for temporary HTTP failures.
        backoff_factor (float): Base backoff factor for retry delays.
    """
    cpus = multiprocessing.cpu_count()
    workers = min(cpus, max_workers) if max_workers else cpus
    Logger.info(f'🚀 Running Data Extraction for Climate Stations using [{workers}] CPUs ...')
    args = [
        (
            station,
            url,
            output_path,
            request_delay_seconds,
            timeout_seconds,
            max_retries,
            backoff_factor
        )
        for station in stations
    ]

    with multiprocessing.Pool(processes=workers) as pool:
        for _ in tqdm(pool.imap(process_station, args), total=len(stations), ascii=True, ncols=75, desc='📤 Extracting'):
            pass
    Logger.info('🟢 Extraction Done.')


def save_climte_station_data(
        station_stats: DataFrame,
        climate_station:int,
        output_path: os.PathLike
    ) -> None:
    """Save the climate station statistics to a CSV file.

    Args:
        station_stats (DataFrame): The DataFrame containing the station statistics.
        climate_station (int): The ID of the climate station.
        output_path (os.PathLike): The directory path where the data will be saved.
    """
    os.makedirs(output_path, exist_ok=True)
    station_stats.to_csv(os.path.join(output_path, str(climate_station)+'.csv'), index=False)
    Logger.debug(f'💾 Station [{climate_station}] statistics stored.')


def process_file(filename: str, root: str) -> pd.DataFrame:
    """Processes a single CSV file and returns a DataFrame with a geometry column."""
    try:
        df = pd.read_csv(os.path.join(root, filename))
        df['geometry'] = [Point(xy) for xy in zip(df.Lon, df.Lat)]
        return df
    except Exception as e:
        Logger.error(f"Error processing file {filename}: {e}")
        return pd.DataFrame()


def generate_geopackage(output_path: os.PathLike) -> None:
    """Generates a GeoPackage file from CSV files located in the specified output path."""
    Logger.info('🛰️  Creating Climate Station Layer ...')
    try:
        buffer = pd.DataFrame()
        all_files = []
        for _, _, files in os.walk(output_path):
            all_files.extend([f for f in files if f.endswith('.csv')])
        with multiprocessing.Pool() as pool:
            results = pool.starmap(process_file, [(f, root) for f in all_files for root, _, _ in os.walk(output_path)])
        buffer = pd.concat(results, ignore_index=True)

        Logger.info('📦 Packaging layer ...')
        geodataframe = gpd.GeoDataFrame(data=buffer, crs="EPSG:4326")
        geodataframe.to_file(
            filename=os.path.join(output_path, 'layer.gpkg'),
            driver='GPKG', layer='climate_stations'
        )
    except Exception as e:
        Logger.error('🔴 ' + str(e))
    Logger.info(f'🟢 Geopackage Done. Stored in {output_path}')


def remove_files_by_extension(
        directory: os.PathLike,
        extension: str = 'csv'
    ) -> None:
    """
    Removes files with a specified extension from the given directory.

    This function lists all files in the specified directory and deletes those that 
    match the provided file extension.

    Args:
        directory (os.PathLike): The path to the directory from which files will be removed.
        extension (str): The file extension to match for removal (e.g., '.csv').
    """
    Logger.info('🧹 Cleaning Up ...')
    for files in glob.iglob(os.path.join(directory, f'*.{extension}')):
        os.remove(files)
