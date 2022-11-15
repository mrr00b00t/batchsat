import os
import json
import numpy as np
import pandas as pd
import requests
import rasterio
import xmltodict
import logging
import geopandas as gpd

from itertools import repeat
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from concurrent.futures import ThreadPoolExecutor


def download_bands(
    account: tuple,
    metadata: dict, uuid: str, polygons: list,
    band: str, resolution: str
) -> list:

    try:
        user_product = metadata['n1:Level-2A_User_Product']
        general_info = user_product['n1:General_Info']
        product_info = general_info['Product_Info']
        product_organisation = product_info['Product_Organisation']
        granule_list = product_organisation['Granule_List']
        granule = granule_list['Granule']
        band_list = granule['IMAGE_FILE']

        band_resolution = '_'.join([band, resolution])
        band_path = next(filter(lambda x: band_resolution in x, band_list)) + '.jp2'

        product_uri = product_info['PRODUCT_URI']
        uri_band_path = f"{product_uri}/{band_path}"
        nodes_uri_band_path = map(lambda x: f"Nodes('{x}')", uri_band_path.split('/'))
        complete_band_path = '/'.join(nodes_uri_band_path)
        
        url = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/{complete_band_path}/$value"

        r = requests.get(url, auth=account, stream=True, allow_redirects=True)

        if r.status_code != 200:
            raise Exception(f'Status code {r.status_code}')

        r.raw.decode_content = True

        bands = list()

        for polygon in polygons:
            with MemoryFile(r.content) as meminfile:
                with meminfile.open(driver='JP2OpenJPEG') as src:
                    wkt = polygon['polygon_envelope']
                    bounds = gpd.GeoSeries.from_wkt([wkt], crs='EPSG:4326').to_crs(src.crs).iloc[0].bounds
                    window = from_bounds(*bounds, transform=src.transform)
                    new_window = Window(
                        max(0., window.col_off-1.),
                        max(0., window.row_off-1.),
                        window.width+2. if window.col_off + window.width+1 < src.meta['width'] else window.width,
                        window.height+2. if window.row_off + window.height+1 < src.meta['height'] else window.height
                    )

                    out_image = src.read(window=new_window)
                    out_transform = src.window_transform(new_window)
                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        'transform': out_transform
                    })

                    band = dict()
                    band['out_image'] = out_image
                    band['out_meta'] = out_meta

                    bands.append(band)

                    logging.info(f"polygon {polygon['polygon_id']} {band_resolution} successfully downloaded")
        
        return bands
    except Exception as e:
        logging.error(
            f'Error while downloading {band}_{resolution}: ' +
            str(e) +
            f" Product {uuid}"
        )
        return list()

def resize_band(band: dict, scale_factor: float, resampling) -> dict:

    with MemoryFile() as memfile:
        with memfile.open(**band['out_meta']) as src:
            src.write(band['out_image'])

            # resample out_image to target shape
            resized_image = src.read(
                out_shape=(
                    src.count,
                    int(src.height * scale_factor),
                    int(src.width * scale_factor)
                ),
                resampling=resampling
            )

            # scale image transform
            resized_transform = src.transform * src.transform.scale(
                (src.width / resized_image.shape[2]),
                (src.height / resized_image.shape[1])
            )

            resized_meta = src.meta
            resized_meta.update({
                "driver": "GTiff",
                'height': resized_image.shape[1],
                'width': resized_image.shape[2],
                'transform': resized_transform
            })

            resized_band = dict()
            resized_band['out_image'] = resized_image
            resized_band['out_meta'] = resized_meta

            return resized_band

def resize_bands(bands: list, scale_factor: float, resampling) -> list:
    seq = map(lambda band: resize_band(band, scale_factor, resampling), bands)
    return list(seq)

def roi_mask(polygon_wkt: str, band: dict) -> dict:

    with MemoryFile() as memfile:
        with memfile.open(**band['out_meta']) as src:
            src.write(band['out_image'])

            gdf = gpd.GeoSeries.from_wkt([polygon_wkt], crs='EPSG:4326').to_crs(src.crs)
            masked, _ = mask(src, gdf, nodata=0)

            return masked

def roi_masks(polygons: list, bands: list) -> list:
    seq = map(lambda polygon, band: roi_mask(polygon['polygon_wkt'], band), polygons, bands)
    return list(seq)

def ROI_quality(SCL: dict, ROI: np.ndarray, minimum_quality: float):
    _SCL = SCL['out_image'].copy()

    # bad classifications
    for i in [1,2,3,7,8,9,10,11]:
        _SCL[_SCL==i] = 0

    # good classifications
    for i in [4,5,6]:
        _SCL[_SCL==i] = 1

    return np.sum(_SCL*ROI) / np.sum(ROI) >= minimum_quality

def ROIs_quality(SCLs: list, ROIs: list, minimum_quality: float) -> list[bool]:
    seq = map(lambda SCL, ROI: ROI_quality(SCL, ROI, minimum_quality), SCLs, ROIs)
    return list(seq)

def save_band(polygon: dict, title: str, band: dict) -> None:

    with rasterio.open(os.path.join('rasters', polygon['polygon_id'], f'{title}.tif'), 'w', **band['out_meta']) as src:
        src.write(band['out_image'])

def save_bands(polygons: list, title: str, bands: list) -> None:
    seq = map(lambda polygon,band: save_band(polygon, title, band), polygons, bands)
    list(seq)

def repeat_list(_list: list, size: int) -> list:

    new_list = []

    while len(new_list) < size:
        new_list += _list

    return new_list[:size]

def main() -> None:

    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv('products-polygons.csv', sep=';', dtype=str)
    df['metadata'] = df['metadata'].apply(lambda x: xmltodict.parse(x))
    df['polygons'] = df['polygons'].apply(lambda x: json.loads(x))

    accs = [('copernicus_user', 'copernicus_password')]
    accounts = repeat_list(accs, len(df))

    with ThreadPoolExecutor(max_workers=len(accs)) as executor:
        seq = executor.map(download_bands, accounts, df.metadata, df.uuid, df.polygons, repeat('SCL'), repeat('20m'))
        df['SCLs'] = [r for r in seq]

        seq = executor.map(resize_bands, df.SCLs, repeat(2), repeat(Resampling.nearest))
        df['SCLs'] = [r for r in seq]

        seq = executor.map(roi_masks, df.polygons, df.SCLs)
        df['ROIs'] = [r for r in seq]

        seq = executor.map(ROIs_quality, df.SCLs, df.ROIs, repeat(0.9))
        df['ROIs_quality'] = [r for r in seq]

        df['polygons'] = df.apply(lambda x: [polygon for (polygon, quality) in zip(x['polygons'], x['ROIs_quality']) if quality], axis=1)
        df['SCLs'] = df.apply(lambda x: [SCL for (SCL, quality) in zip(x['SCLs'], x['ROIs_quality']) if quality], axis=1)
        df['ROIs'] = df.apply(lambda x: [ROI for (ROI, quality) in zip(x['ROIs'], x['ROIs_quality']) if quality], axis=1)

        if not os.path.exists('rasters'):
            os.makedirs('rasters')

        _ = df['polygons'].apply(
            lambda polygons: [
                os.makedirs(os.path.join('rasters', polygon['polygon_id']))
                for polygon in polygons if not os.path.exists(os.path.join('rasters', polygon['polygon_id']))
        ])

        seq = executor.map(download_bands, accounts, df.metadata, df.uuid, df.polygons, repeat('TCI'), repeat('10m'))
        df['TCIs'] = list(seq)

        seq = executor.map(save_bands, df.polygons, df.title, df.TCIs)
        _ = list(seq)

if __name__ == '__main__':
    main()
