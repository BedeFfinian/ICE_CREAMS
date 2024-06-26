#!/usr/bin/env python3
# coding: utf-8
"""
Apply ICE CREAMS to Masked Sentinel-2.

Takes a trained Neural Network model (saved with pickle) and runs this over a S2 image
in SAFE format using xarray.

Author: Bede Ffinian Rowe Davies 
Date: 2023-03-30 edited 2024-06-23

"""
import argparse
import glob
import os

import numpy

import geopandas
import xarray
import rasterio
import rioxarray
from dask.diagnostics import ProgressBar
from fastai.tabular.all import load_learner

DEFAULT_FASTAI_MODEL = os.path.join(
    os.path.dirname(__file__), "ICECREAMS_V1_1.pkl"
)

CLASSES_NUMBER_ID_DICT = {
    1: "Bare Sand",
    2: "Bare Sediment",
    3: "Chlorophyta",
    4: "Magnoliopsida",
    5: "Microphytobenthos",
    6: "Phaeophyceae",
    7: "Rhodophyta",
    8: "Water",
    9: "Xanthophyceae",
}


def build_s2_mask_scl_mask(scl_data):
    """
    Takes Sentinel-2 SCL image as an xarray data array and returns a mask based on flags (True=valid data)

    S2 flags are:

    1: saturated or defective
    2: dark area pixels
    3: cloud shadows
    4: vegetation
    5: not vegetated
    6: water
    7: unclassified
    8: cloud medium probability
    9: cloud high probability
    10: thin cirrus
    11: snow

    Parameters
    ----------
    ds : xarray.Dataset with scl (i.e., cloud mask) variable.
    """
    mask = xarray.where(
        (scl_data == 1)
        | (scl_data == 11),
        True,
        False,
    )

    return mask


def _get_s2_files_from_safe(input_s2_safe) -> dict:
    """ "
    Function to get the required jp2 files for a scene from within a .SAFE file
    """
    output_files_dict = {}

    # Find jp2 files
    output_files_dict["b01_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B01_60m.jp2")
    )[0]
    output_files_dict["b02_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B02_10m.jp2")
    )[0]
    output_files_dict["b03_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B03_10m.jp2")
    )[0]
    output_files_dict["b04_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B04_10m.jp2")
    )[0]
    output_files_dict["b05_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B05_20m.jp2")
    )[0]
    output_files_dict["b06_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B06_20m.jp2")
    )[0]
    output_files_dict["b07_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B07_20m.jp2")
    )[0]
    output_files_dict["b08_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B08_10m.jp2")
    )[0]
    output_files_dict["b08a_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B8A_20m.jp2")
    )[0]
    output_files_dict["b09_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B09_60m.jp2")
    )[0]
    output_files_dict["b11_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B11_20m.jp2")
    )[0]
    output_files_dict["b12_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*B12_20m.jp2")
    )[0]
    output_files_dict["scl_file"] = glob.glob(
        os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", "*SCL_20m.jp2")
    )[0]
    
    output_files_dict["processing_base_line"] = int(os.path.basename(input_s2_safe.rstrip("/")).split("_")[3][1:5])

    return output_files_dict


def _read_s2_data_xarray(input_s2_files, mask_vector_file=None):
    """
    Read s2 data to an xarray object, masking if a mask is provided.

    Takes a dictionary of input files, which can be anything rioxarray can open
     (e.g., .jp2 file, S3 bucket).
    """

    # Read to xarray DataArray
    b01 = rioxarray.open_rasterio(
        input_s2_files["b01_file"], chunks={"x": 512, "y": 512}
    )
    b02 = rioxarray.open_rasterio(
        input_s2_files["b02_file"], chunks={"x": 512, "y": 512}
    )
    b03 = rioxarray.open_rasterio(
        input_s2_files["b03_file"], chunks={"x": 512, "y": 512}
    )
    b04 = rioxarray.open_rasterio(
        input_s2_files["b04_file"], chunks={"x": 512, "y": 512}
    )
    b05 = rioxarray.open_rasterio(
        input_s2_files["b05_file"], chunks={"x": 512, "y": 512}
    )
    b06 = rioxarray.open_rasterio(
        input_s2_files["b06_file"], chunks={"x": 512, "y": 512}
    )
    b07 = rioxarray.open_rasterio(
        input_s2_files["b07_file"], chunks={"x": 512, "y": 512}
    )
    b08 = rioxarray.open_rasterio(
        input_s2_files["b08_file"], chunks={"x": 512, "y": 512}
    )
    b08a = rioxarray.open_rasterio(
        input_s2_files["b08a_file"], chunks={"x": 512, "y": 512}
    )
    b09 = rioxarray.open_rasterio(
        input_s2_files["b09_file"], chunks={"x": 512, "y": 512}
    )
    b11 = rioxarray.open_rasterio(
        input_s2_files["b11_file"], chunks={"x": 512, "y": 512}
    )
    b12 = rioxarray.open_rasterio(
        input_s2_files["b12_file"], chunks={"x": 512, "y": 512}
    )
    scl = rioxarray.open_rasterio(
        input_s2_files["scl_file"], chunks={"x": 512, "y": 512}
    )


    # Bias removed for images after 2022 using baseline processor after 400
    
    if input_s2_files["processing_base_line"] > 399:
        for band in [b01, b02, b03, b04, b05, b06, b07, b08, b08a, b09, b11, b12]:
            band.data = band.data - 1000

    
    # Resample 20m and 60m bands to 10m
    b01_10m = b01.interp(x=b02.x, y=b02.y, method="nearest")
    b05_10m = b05.interp(x=b02.x, y=b02.y, method="nearest")
    b06_10m = b06.interp(x=b02.x, y=b02.y, method="nearest")
    b07_10m = b07.interp(x=b02.x, y=b02.y, method="nearest")
    b08a_10m = b08a.interp(x=b02.x, y=b02.y, method="nearest")
    b11_10m = b11.interp(x=b02.x, y=b02.y, method="nearest")
    b12_10m = b12.interp(x=b02.x, y=b02.y, method="nearest")
    b09_10m = b09.interp(x=b02.x, y=b02.y, method="nearest")
    scl_10m = scl.interp(x=b02.x, y=b02.y, method="nearest")

    # Save all to one Raw dataset
    s2_data_raw = xarray.Dataset(
        {
            "Reflectance_B01": b01_10m,
            "Reflectance_B02": b02,
            "Reflectance_B03": b03,
            "Reflectance_B04": b04,
            "Reflectance_B05": b05_10m,
            "Reflectance_B06": b06_10m,
            "Reflectance_B07": b07_10m,
            "Reflectance_B08": b08,
            "Reflectance_B8A": b08a_10m,
            "Reflectance_B09": b09_10m,
            "Reflectance_B11": b11_10m,
            "Reflectance_B12": b12_10m,
            "SCL": scl_10m,
        }
    )
    # Set CRS
    s2_data_raw.rio.set_crs(b02.rio.crs)

    # Apply SCL mask to data
    scl_mask = build_s2_mask_scl_mask(scl_10m)
    s2_data_raw = s2_data_raw.where(~scl_mask)

    if mask_vector_file is not None:
        print(f"Masking S2 scene to {mask_vector_file}")
        # First subset
        mask_vector = geopandas.read_file(mask_vector_file)
        x_min = float(mask_vector.bounds["minx"].min())
        x_max = float(mask_vector.bounds["maxx"].max())
        y_min = float(mask_vector.bounds["miny"].min())
        y_max = float(mask_vector.bounds["maxy"].max())
        s2_data_raw = s2_data_raw.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

        # Then apply mask
        mask_raster = rasterio.features.geometry_mask(
            mask_vector.geometry,
            out_shape=s2_data_raw.Reflectance_B01.shape[1:],
            transform=s2_data_raw.rio.transform(recalc=True),
        )
        s2_data_raw = s2_data_raw.where(~mask_raster)
        s2_data_raw["study_site"] = xarray.DataArray(
            data=numpy.expand_dims(mask_raster, axis=0),
            dims=s2_data_raw.dims,
            coords=s2_data_raw.coords,
        )

    return s2_data_raw


def read_s2_safe(input_s2_safe, mask_vector_file=None):
    """
    Function to read S2 data from SAFE file and return as a Dataset with
    bands resampled to 10m
    """
    s2_files_dict = _get_s2_files_from_safe(input_s2_safe)
    return _read_s2_data_xarray(s2_files_dict, mask_vector_file)

def standerdise_reflectance(s2_data_raw):
    """
    Standardise reflectance by scaling from 0 - 1 where 1 is the maximum value for each
    band of a pixel.

    Returns as a separate xarray

    """
    data_vars_list = []
    # Go through each data variable
    for data_var in s2_data_raw.data_vars:
        # Don't mask study site
        if data_var in ["study_site", "SCL"]:
            continue
        else:
            data_vars_list.append(data_var)

    # Calculate the min and max, xarray will ignore no data values
    s2_data_raw_array = s2_data_raw[data_vars_list].to_array(dim="wavelength")
    var_min = s2_data_raw_array.min(dim="wavelength")
    var_max = s2_data_raw_array.max(dim="wavelength")

    s2_data_standardised = (s2_data_raw[data_vars_list] - var_min) / (var_max - var_min)

    update_names = {
        data_var: data_var.replace("Reflectance_", "Reflectance_Stan_")
        for data_var in data_vars_list
    }

    # Rename variables
    s2_data_standardised = s2_data_standardised.rename_vars(update_names)

    return s2_data_standardised


def calc_ndvi_true(s2_data_raw):
    """
    Function to calculate NDVI from raw S2 data loaded as xarray Dataset

    Returns xarray DataArray
    """
    red_raw = s2_data_raw["Reflectance_B04"]
    nir_raw = s2_data_raw["Reflectance_B08"]

    ndvi_raw = (nir_raw - red_raw) / (nir_raw + red_raw)
    ndvi_raw.name = "NDVI"

    return ndvi_raw


def calc_ndwi(s2_data_raw):
    """
    Function to calculate NDWI from S2 data loaded as xarray Dataset

    Returns xarray DataArray
    """
    green_raw = s2_data_raw["Reflectance_B03"]
    nir_raw = s2_data_raw["Reflectance_B08"]

    ndwi_raw = (green_raw - nir_raw) / (nir_raw + green_raw)
    ndwi_raw.name = "NDWI"

    return ndwi_raw


def calc_spc(s2_data_raw):
    """
    Function to calculate Seagrass Cover from S2 data loaded as xarray Dataset For Post-Processing

    Returns xarray DataArray
    """
    red_raw = s2_data_raw["Reflectance_B04"]
    nir_raw = s2_data_raw["Reflectance_B08"]
    NDVI = (nir_raw - red_raw) / (nir_raw + red_raw)
    spc = 172.06*NDVI-22.18
    spc.name = "SPC"

    return spc


def apply_classification(input_xarray, class_model):
    """
    Apply classification to a list of input xarray DataArrays
    """
    # Convert to pandas dataframe
    input_df = input_xarray.to_dataframe()

    # Fill nan values with 0
    input_df = input_df.fillna(0)

    # Apply model with a batch size of 4096
    Out_Class_dl = class_model.dls.test_dl(input_df, bs=4096)
    preds, _ = class_model.get_preds(
        dl=Out_Class_dl
    )  # This creates a tensor with 9 prediction classes 0:8

    Out_Class = preds.argmax(axis=1)  # This pulls the majority class
    Class_Probs = preds.max(
        axis=1
    )  # These lines pull the probability the model gave the majority class
    Class_Probs_values = (
        Class_Probs.values
    ) 

    input_df["Out_Class"] = Out_Class
    input_df["Class_Probs"] = Class_Probs_values

    input_df["SPC"] =  input_df["SPC"].where(input_df["Out_Class"] == 3, other=0)
    input_df["SPC"] = input_df["SPC"].where(input_df["SPC"]>=0,other=0)
    input_df["SPC"] = input_df["SPC"].where(input_df["SPC"]<=100,other=100)

    input_df["SPC20"] =  input_df["SPC"].where(input_df["Out_Class"] == 3, other=0)
    input_df["SPC20"] = input_df["SPC20"].where(input_df["SPC20"]>=20,other=0)
    input_df["SPC20"] = input_df["SPC20"].where(input_df["SPC20"]<=100,other=100)
    
    input_df["Seagrass_Cover"] = input_df["SPC20"]
    
    # Make sure data masked out in input are also masked out in classification
    input_df["Out_Class"] = input_df["Out_Class"].where(
        input_df["Reflectance_B01"] != 0, other=-1
    )
    input_df["Class_Probs"] = input_df["Class_Probs"].where(
        input_df["Reflectance_B01"] != 0, other=-1
    )

    input_df["Seagrass_Cover"] = input_df["Seagrass_Cover"].where(
        input_df["Reflectance_B01"] != 0, other=-1
    )
    
    # Add 1 to be consistent with R output
    input_df["Out_Class"] = input_df["Out_Class"] + 1

    # Convert back to xarray
    return input_df.to_xarray()


def classify_s2_scene(
    input_s2_safe,
    output_gtiff,
    saved_model,
    mask_vector_file=None,
    debug=False,
):
    """
    Function to classify an S2 scene netCDF file

    """
    class_model = load_learner(saved_model)

    # Open scene into xarray dataset
    # Specify chunksize so uses dask and doesn't load all data to RAM
    print(f"Reading in data from {input_s2_safe}...")
    s2_data_raw = read_s2_safe(input_s2_safe, mask_vector_file)

    # Standardise data
    s2_data_stan = standerdise_reflectance(s2_data_raw)

    # Calculate NDVI, NDWI and SPC
    ndwi_raw = calc_ndwi(s2_data_raw)
        
    ndvi_true_raw = calc_ndvi_true(s2_data_raw)
    
    spc_raw = calc_spc(s2_data_raw)
    
    
    # Merge to a single xarray
    s2_data = xarray.merge([s2_data_raw, s2_data_stan, ndwi_raw, ndvi_true_raw,spc_raw])

    # Apply classification. Will print progress
    print("Performing classification")
    Out_Classified_s2_data = apply_classification(s2_data, class_model)

    # Set up output dataset
    # If running in debug mode don't subset and write out all variables
    if debug:
        Out_Class_dataset = Out_Classified_s2_data.squeeze(dim="band", drop=True)
    else:
        Out_Class_dataset = Out_Classified_s2_data[
            ["Out_Class", "Class_Probs","Seagrass_Cover"]
        ].squeeze(dim="band", drop=True)

    Out_Class_dataset.assign_attrs(
        {
            "description": "ICE CREAMS Model Output",
            "class_ids": str(CLASSES_NUMBER_ID_DICT),
        }
    )
    Out_Class_dataset.rio.set_crs(s2_data.rio.crs)

    ## Write out to Geotiff
    print("Writing out")
    with ProgressBar():
        Out_Class_dataset.rio.to_raster(
            output_gtiff, driver="COG", tiled=True, windowed=True, dtype=numpy.float32
        )

    print(f"Saved to {output_gtiff}")
    return output_gtiff


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Apply ICE CREAMS to an S2 image"
    )
    parser.add_argument(
        "insafe", help="Input S2 file in SAFE format (Atmospherically Corrected)"
    )
    parser.add_argument("outfile", help="Output file for classification")
    parser.add_argument(
        "--mask",
        required=False,
        default=None,
        help="Vector file specifying the bounds to run classification within. Will mask out areas outside polygon",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=DEFAULT_FASTAI_MODEL,
        help="Fastai saved model file",
    )
    parser.add_argument(
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Debug mode, writes out all layers to output file and prints more output",
    )
    args = parser.parse_args()

    classify_s2_scene(
        args.insafe,
        args.outfile,
        args.model,
        args.mask,
        debug=args.debug,
    )
