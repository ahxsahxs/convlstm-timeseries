import numpy as np
import xarray
from pathlib import Path
import logging
import pandas as pd


def load_tile(tile_path: Path):
    '''
    Return dict of xarray datasets with dimensions:
        doy: list of length 30 containing the day of year for each sentinel 2 observation
        s2: {'time': 30, 'lat': 128, 'lon': 128} (4 spectral bands: s2_B02, s2_B03, s2_B04, s2_B8A)
        cloudmask: {'time': 30, 'lat': 128, 'lon': 128} (single binary band: s2_mask, 1 if cloudy)
        weather: {'time': 150} (6 time series across 150 days: eobs_rr, eobs_pp, eobs_qq, eobs_tg, eobs_tn, eobs_tx)
        dem: {'lat': 128, 'lon': 128} (3 DEM bands: nasa_dem, cop_dem, alos_dem)
        lulc: {'lat': 128, 'lon': 128} (1 landcover categorical band: esawc_lc)
        geomorph: {'lat': 128, 'lon': 128} (1 geomorphology categorical band: geom_cls)
    '''

    tile = xarray.load_dataset(tile_path)

    mask_name = "s2_mask" if "s2_mask" in tile.data_vars else "s2_dlmask"

    s2_data = tile[["s2_B02", "s2_B03", "s2_B04", "s2_B8A"]].isel(time=slice(4, None, 5))
    cloudmask_data = tile[[mask_name]].isel(time=slice(4, None, 5))
    s2_doy = list(pd.to_datetime(tile["time"][4::5]).day_of_year)
    weather_data = tile[["eobs_hu", "eobs_rr", "eobs_pp", "eobs_qq", "eobs_tg", "eobs_tn", "eobs_tx"]]
    dem_data = tile[["nasa_dem", "cop_dem", "alos_dem"]]
    lulc_data = tile[["esawc_lc"]]
    geomorph_data = tile[["geom_cls"]]


    return {
        "doy": s2_doy,
        "s2": s2_data,
        "cloudmask": cloudmask_data,
        "weather": weather_data,
        "dem": dem_data,
        "lulc": lulc_data,
        "geomorph": geomorph_data,
    }


if __name__ == "__main__":
    train_example = Path("greenearthnet/train/29SND/29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.nc")
    
    tile_data = load_tile(train_example)

    for key, val in tile_data.items():
        if key == "doy":
            print("Day of the Year: ", val)
        else:
            print(f"Shape of {key}: {val.dims}")

    val_example = Path("greenearthnet/val_chopped/JAS20/minicube_0_29SND_39.29_-8.56.nc")
    tile_data = load_tile(val_example)