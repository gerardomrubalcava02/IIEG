# ==============================================================================
# Project Name: Extracting Climatological Statistical Information Application
# Script Name: extractor.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   The application allows users to consult and extract historical information 
#   from the conventional weather stations that make up the National Network 
#   of CONAGUA.
#
# License: MIT
# ==============================================================================

import os
import re
import requests
import pandas as pd

from dataclasses import dataclass, field
from io import StringIO
from unidecode import unidecode
from utils.logger import Logger
from core.patterns import Pattern
from pandas import DataFrame

SUMMARY_POSITION = 3
SUMMARY_STATS_POSITION = 4
STATS_TITLE_POSITION = 0
STATS_HEADER_POSITION = 1

pattern = Pattern()


@dataclass
class Summary:
    """Data class to hold summary information for a climate station."""
    name: str
    municipality: str
    lat: float
    lon: float
    height: int
    status: str


@dataclass
class Statistic:
    """Data class to hold statistical data for a climate station."""
    title: str
    statistics: DataFrame = field(repr=False)


@dataclass
class Extraction:
    """Data class to hold the extraction results."""
    summary: Summary
    statistics: list[Statistic]

type StationResult = list[DataFrame]


@dataclass
class ClimateExtractor:
    """
    Extract climate data from a specified station.
    """
    station: str
    url: str
    result: StationResult = field(init=False, default=None)

    def run(self) -> None:
        """
        Execute the extraction process for the station data.

        This method orchestrates the entire extraction process by first 
        retrieving the station attributes (state and number). It checks 
        if the station is valid and logs a warning if it is not. Then, 
        it makes an HTTP request to fetch the station data. If the request 
        fails or the data cannot be processed, it logs appropriate warnings. 
        If the data is successfully retrieved, it proceeds to parse and 
        transform the data, storing the result in the instance's result attribute.
        """
        station_state, station_number, valid_station = self._get_station_attributes()
        if not valid_station:
            Logger.warning(f'🟡 Invalid Station name [{self.station}]')
        data, request_code = self._request(
            station_state=station_state, station_number=station_number
        )
        if not data:
            Logger.warning(f'🟡 Unable to process request in station [{self.station}] HTTP code {request_code}')
            return
        Logger.debug(f'⚙️ Processing station [{self.station}]')
        parsed = self._parse(data=data)
        if not parsed:
            Logger.warning(f'🟡 Unable to parse stats of station [{self.station}]')
            return
        self.result = self._transform(data=parsed)

    def _request(self, station_state: str, station_number: str) -> str | None:
        """
        Make an HTTP GET request to fetch data for the specified station.

        This function constructs a URL using the provided station state 
        and station number, then performs an HTTP GET request to retrieve 
        the corresponding data. If the request is successful (HTTP status 
        code 200), it returns the content of the response. If the request 
        fails, it returns None.

        Args:
            station_state (str): The state of the station to include in the request URL.
            station_number (str): The number of the station to include in the request URL.

        Returns:
            str | None: The content of the response if the request is successful, 
                        otherwise None.
        """
        content = None
        formated_url = os.path.join(self.url, station_state, 'mes'+station_number+'.txt')
        response = requests.get(formated_url)
        if response.status_code == 200:
            content = response.text
        return content, response.status_code

    def _get_station_attributes(self) -> tuple[str | None, str | None, bool]:
        """
        Extracts attributes from the station string.

        This function searches for the station state and station number 
        within the instance's station attribute using regular expressions. 
        It returns a tuple containing the station state, station number, 
        and a boolean indicating whether both attributes were successfully 
        found.

        Returns:
            tuple[str | None, str | None, bool]: A tuple containing:
                - station_state (str | None): The state of the station if found, otherwise None.
                - station_number (str | None): The number of the station if found, otherwise None.
                - is_valid_station (bool): True if both station state and number are found, otherwise False.
        """
        station_state = None
        station_number = None
        is_valid_station = True
        station_state_search = re.search(pattern=pattern.station_state, string=self.station)
        if station_state_search:
            station_state = station_state_search.group()
        station_number_search = re.search(pattern=pattern.station_number, string=self.station)
        if station_number_search:
            station_number = station_number_search.group()
        if not station_state or not station_number:
            is_valid_station = False
        return station_state, station_number, is_valid_station

    def _parse(self, data: str | None) -> Extraction | None:
        """Parse the fetched data into a structured format.

        Args:
            data (str | None): The raw data fetched from the HTTP request, or None if no data is available.

        Returns:
            Extraction | None: An instance of the Extraction data class containing structured summary and statistics,
                            or None if the input data is not valid.
        """
        extraction = None
        if data:
            paragraphs = data.split('\n\n')
            extract = paragraphs[SUMMARY_POSITION]
            summary = self._get_summary(paragraph=extract)
            stats = self._get_stats(paragraphs[SUMMARY_STATS_POSITION:])
            extraction = Extraction(summary=summary, statistics=stats)
        return extraction

    def _transform(self, data: Extraction) -> DataFrame:
        """Transform the extracted data into a DataFrame.

        Args:
            data (Extraction): The extracted data containing summary and statistics.

        Returns:
            DataFrame: A DataFrame containing the transformed records with additional context fields.
        """
        records = pd.DataFrame()
        if data:
            statistics = data.statistics
            summary = data.summary
            for statistic in statistics:
                df = statistic.statistics
                # Adding new fields
                df['Índice'] = statistic.title
                df['Estación'] = self.station
                df['Nombre'] = summary.name
                df['Municipio'] = summary.municipality
                df['Lat'] = summary.lat
                df['Lon'] = summary.lon
                df['Altura'] = summary.height
                df['Situación'] = summary.status
                if records.empty:
                    records = df
                else:
                    records = pd.concat([records, df], ignore_index=True)
        return records

    def get(self):
        return self.result

    def _get_summary(self, paragraph: str) -> Summary:
        """Extract summary information from a given paragraph.

        Args:
            paragraph (str): The paragraph containing summary data.

        Returns:
            Summary: An instance of the Summary data class containing extracted information.
        """
        name = self._get_field(data=paragraph, pattern=pattern.station_name)
        municipality = self._get_field(data=paragraph, pattern=pattern.municipality_name)
        lat = self._get_field(data=paragraph, pattern=pattern.lat)
        lon = self._get_field(data=paragraph, pattern=pattern.lon)
        height = self._get_field(data=paragraph, pattern=pattern.height)
        status = self._get_field(data=paragraph, pattern=pattern.station_status)
        summary = Summary(
            name=name,
            municipality=municipality,
            lat=float(lat),
            lon=float(lon),
            height=int(height),
            status=status
        )
        return summary

    def _get_stats(self, paragraphs: list[str]) -> list[Statistic]:
        """Extract statistical data from a list of paragraphs.

        Args:
            paragraphs (list[str]): A list of paragraphs containing statistical data.

        Returns:
            list[Statistic]: A list of Statistic data class instances containing extracted statistics.
        """
        stats = list()
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            rows = paragraph.split('\n')
            title = rows[STATS_TITLE_POSITION]
            data = '\n'.join(rows[STATS_HEADER_POSITION:-4])
            df = pd.read_csv(StringIO(data), sep='\t')
            statistic = Statistic(title=title, statistics=df)
            stats.append(statistic)
        return stats

    def _get_field(self, data: str, pattern: str) -> str | None:
        """Extract a specific field from the data using a regex pattern.

        Args:
            data (str): The string data to search within.
            pattern (str): The regex pattern to use for extraction.

        Returns:
            str | None: The extracted field value or None if not found.
        """
        field = None
        data = unidecode(data)
        scan = re.search(pattern=pattern, string=data)
        if scan:
            extraction = scan.group()
            if extraction:
                field = extraction.split(':')[-1]
                field = field.strip()
        return field
