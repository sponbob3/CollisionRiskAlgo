# Datasets

Each dataset is one subfolder here, named `<ICAO>_<label>`, holding one
file per day of OpenSky-style ADS-B data (`.parquet` preferred, `.csv`
accepted; when both exist for the same day, the parquet wins):

```
datasets/
  KDAB_2025/            <- airport KDAB, label "2025"
    ..._20250102_raw.parquet
    ..._20250103_raw.parquet
  KBNA_spring26/        <- airport KBNA, label "spring26"
    ...
```

The folder-name prefix before the first underscore selects the airport
profile `airports/<ICAO>.yaml` (generate a new one with
`python run_proximity.py new-airport <ICAO>`).

Required columns in every daily file: `timestamp`, `icao24`, `callsign`,
`latitude`, `longitude`, `altitude`, `geoaltitude`, `vertical_rate`,
`groundspeed`, `track`, `onground`.

Run a dataset with:

```bash
python run_proximity.py KDAB_2025
```

Each run writes fresh numbered folders under `output/<ICAO>/goaround/run_NN/`
and `output/<ICAO>/proximity_risk/run_NN/`.

Raw data is large (several GB per year) and gitignored — nothing in this
folder except this README is ever committed.
