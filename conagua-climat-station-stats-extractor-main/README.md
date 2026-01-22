# Application for Extracting Climatological Statistical Information 🤖

<img src="logo.svg" width="512"/>

![build succeeded](https://img.shields.io/badge/Application-Geo-blue.svg) ![build succeeded](https://img.shields.io/badge/Version-1.0.0-yellow.svg)  ![build succeeded](https://img.shields.io/badge/Python-3.12+-brightgreen.svg) ![build succeeded](https://img.shields.io/badge/License-MIT-purple.svg)

## 📖 Overview

The application (`climstats`) allows users to consult and extract historical information from the conventional weather stations that make up the National Network of [CONAGUA](https://smn.conagua.gob.mx/es/climatologia/informacion-climatologica/informacion-estadistica-climatologica). This information ranges from the first recorded data to the most recent data available in the SMN databases reported by the Basin Organizations and Local Directorates of CONAGUA.

Information is available from just over 5,400 weather stations, of which approximately 2,800 report data, including 126 from Jalisco, while the rest have temporarily ceased operations or no longer exist; however, they maintain a considerable archive of information. The main climatic variables that can be consulted are: extreme temperatures (maximum and minimum), accumulated precipitation over 24 hours, some phenomena such as thunderstorms, fog, hail, sky coverage, evaporation, and climatological normals.

As a result, the application generates a GeoPackage (GPKG) file that contains the historical data for the identified climate stations.

## 📦 Requirements

- Python 3.12 or higher
- Python environment like: venv, conda, mamba, etc.
- Required libraries:
  - `requests`
  - `unidecode`
  - `geopandas`

## 🛠️ Configuration

To set up the application, follow these steps:

1. Clone the repository and navigate to the project directory:
```bash
git clone https://iieg-app.jalisco.gob.mx/iieg/conagua-climat-station-stats-extractor.git
cd conagua-climat-station-stats-extractor
```

2. Install requirements, use:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

> 💡 You can modify the `stations.yaml` or `jalisco_stations.yaml` files if you prefer using a configuration file from the `configs` directory.

#### Extract and Save Climate Station Statistics

```bash
python climstats.py extract --config configs/jalisco_stations.yaml
```

> 💡 The application generates a GPKG file, which is stored in the output directory specified in the configuration file.

#### Tuning Request Concurrency

If CONAGUA is rate-limiting your requests, lower the number of workers and add a delay between requests in your config file:

```yaml
max_workers: 2
request_delay_seconds: 1.0
timeout_seconds: 30
max_retries: 5
backoff_factor: 1.0
```

These settings reduce simultaneous requests and apply retries with exponential backoff to handle temporary blocks.

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue for any suggestions or improvements.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
