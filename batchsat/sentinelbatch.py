import os
import time
import json
import logging
import argparse
import requests
import pandas as pd
import shapely.wkt as wkt
import rasterio
import xmltodict

import geopandas as gpd
from datetime import date
from functools import reduce
from itertools import repeat
from sentinelsat import SentinelAPI
from concurrent.futures import ThreadPoolExecutor

from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.enums import Resampling as Rs
from rasterio.windows import Window, from_bounds


rows_from_iterrows = lambda x: x[1]
concat_dfs = lambda x, y: pd.concat([x, y], ignore_index=True, axis=1)
is_polygon_inside = lambda x: wkt.loads(x['footprint']).contains(wkt.loads(x['polygon']['polygon_envelope']))

def repeat_list(_list: list, size: int):

    for i in range(size):
        yield _list[i % len(_list)]

class SentinelBatch:
    def __init__(self, accounts: list, cloudcoverpercentage: tuple,
        platformname: str = 'Sentinel-2',
        producttype: str = 'S2MSI2A', processinglevel = 'Level-2A',
        sleep_time: float = 0.125
    ) -> None:

        self.accounts = accounts
        self.platformname = platformname
        self.producttype = producttype
        self.processinglevel = processinglevel
        self.sleep_time = sleep_time
        self.cloudcoverpercentage = cloudcoverpercentage

        logging.basicConfig(level=logging.INFO)

    
    def polygon_time_series(self, polygons_df: pd.DataFrame, roi_visibility: float,
        band: str, resolution: str, max_workers: int = 10, ignore_offline: bool = True
    ) -> pd.DataFrame:

        assert band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'AOT', 'WVP', 'SCL', 'TCI']
        assert resolution in ['10m', '20m', '60m'], 'resolution not supported'
        assert .0 <= roi_visibility <= 1., 'visibility of region of interest must be in [0, 1]'
        assert ignore_offline in [False, True], 'ignore offline products must be a bool value'

        bands_download_resolution: dict = {
            'B02': '10m', 'B03': '10m', 'B04': '10m', 'B08': '10m', 'TCI': '10m', 'AOT': '10m', 'WVP': '10m',
            'B05': '20m', 'B06': '20m', 'B07': '20m', 'B8A': '20m', 'B11': '20m', 'B12': '20m', 'SCL': '20m',
            'B01': '60m', 'B09': '60m'
        }

        self.roi_visibility = roi_visibility
        self.band = band
        self.resolution = resolution
        self.download_resolution = bands_download_resolution[band]
        self.scale_factor = float(self.resolution.replace('m', '')) / float(self.download_resolution.replace('m', ''))
        self.resampling = Rs.nearest if band == 'SCL' else Rs.cubic
        self.ignore_offline = ignore_offline
        
        df = polygons_df.copy()[:10]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            # get all the products and filter products that not fully contain its polygon
            df['account'] = list(repeat_list(self.accounts, df.shape[0]))
            seq = df.iterrows()
            seq = map(rows_from_iterrows, seq)
            seq = executor.map(self.products_from_polygons, seq)

            df = reduce(concat_dfs, seq)
            df = df[df.apply(is_polygon_inside, axis=1)]

            size_before = df.shape[0]

            # group polygons by products, they might have repeated polygons per product
            df  = df.groupby(
                ['uuid', 'title', 'link', 'link_alternative', 'summary', 'footprint', 'filename', 'identifier', 'ondemand'],
                as_index=False
            ).agg({'polygon': lambda x: list(x)}).rename(columns={'polygon': 'polygons'})
            
            size_after = df.shape[0]

            reduction = round(100*(1. - size_after/size_before), 2)
            logging.info(
                f"Processing {size_after} products. " +
                f"Reusing products that contains one or more polygons (Reduction: {reduction}%)"
            )
            
            df['account'] = list(repeat_list(self.accounts, df.shape[0]))
            seq = df.iterrows()
            seq = map(rows_from_iterrows, seq)
            seq = executor.map(self._from_row_to_band_resized, seq)
            df = pd.DataFrame(seq)
            df = df.query('online == \'true\' and metadata != \'error\'')

            df.to_csv('products-polygons.csv', sep=';', index=False)

    def _from_row_to_band_resized(self, row: pd.Series) -> pd.Series:

        row = self.product_online_status(row)
        row = self.trigger_online(row)
        row = self.product_metadata(row)
        row = self.download_rasters(row)
        row = self.resize_rasters(row)
        row = self.save_rasters(row)
        
        return row

    def products_from_polygons(self, row: pd.Series) -> pd.DataFrame:

        account = row['account']
        api = SentinelAPI(account[0], account[1], 'https://apihub.copernicus.eu/apihub')

        polygon_id = row['polygon_id']
        polygon_wkt = row['polygon_wkt']
        dt_from = row['dt_from'].replace('-', '')
        dt_to = row['dt_to'].replace('-', '')
        
        try:
            time.sleep(self.sleep_time)

            polygon_envelope = wkt.loads(polygon_wkt).envelope.wkt

            query_result = api.query(
                polygon_envelope,
                date=(dt_from, dt_to),
                platformname=self.platformname,
                producttype=self.producttype,
                processinglevel=self.processinglevel,
                cloudcoverpercentage=self.cloudcoverpercentage
            )
            
            products = api.to_dataframe(query_result)
            
            products['polygon'] = products['uuid'].apply(lambda _ : {
                'polygon_id': polygon_id,
                'polygon_wkt': polygon_wkt,
                'polygon_envelope': polygon_envelope
            })

            logging.info(f"Retrieved {len(products)} products from polygon {polygon_id}")
        except Exception as e:
            logging.error(
                'Error while retrieving products: ' +
                str(e) +
                f" polygon {polygon_id}"
            )
            products = pd.DataFrame()
        
        return products

    def product_online_status(self, row: pd.Series) -> pd.Series:
        
        uuid = row['uuid']
        url: str = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/Online/$value"
        status: str | None = None

        try:
            time.sleep(self.sleep_time)

            account = row['account']
            r = requests.get(url, auth=account)
            
            if r.status_code != 200:
                raise Exception(f'Status code {r.status_code}')

            status = r.content.decode('utf-8')

            logging.info(f"{uuid} is online? -> {status}")
        except Exception as e:
            logging.error(
                'Error while checking online: ' +
                str(e) +
                f" Product {url}"
            )
            status = 'error'
        
        row['online'] = status

        return row

    def trigger_online(self, row: pd.Series) -> pd.Series:

        online = row['online']
        if online == 'true':
            row['triggered'] = 'false'
            return row

        uuid = row['uuid']
        url: str = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/$value"
        triggered: str | None = None

        try:
            time.sleep(self.sleep_time)

            account = row['account']
            r = requests.get(url, auth=account, allow_redirects=True)

            if r.status_code != 202:
                raise Exception(f'Status code {r.status_code}')

            triggered = 'true'
            logging.info(f"Product '{uuid}' successfully triggered to online, try to download in a few hours")
        except Exception as e:
            logging.error(
                'Error while triggering offline product: ' +
                str(e) +
                f" Product {url}"
            )
            triggered = 'error'
        
        row['triggered'] = triggered
        return row

    def product_metadata(self, row: pd.Series) -> pd.Series:

        online = row['online']
        if online != 'true':
            row['metadata'] = 'error'
            return row

        uuid = row['uuid']
        title = row['title']
        url: str = f"https://apihub.copernicus.eu/apihub/odata/v1/Products('{uuid}')/Nodes('{title}.SAFE')/Nodes('MTD_MSIL2A.xml')/$value"
        metadata: str | None = None

        try:
            time.sleep(self.sleep_time)

            account = row['account']
            r = requests.get(url, auth=account)

            if r.status_code != 200:
                raise Exception(f'Status code {r.status_code}')

            metadata = r.content.decode('utf-8').replace('\n', '')

            logging.info(f"Product {uuid} metadata successfully retrieved")
        except Exception as e:
            logging.error(
                'Error while retrieving metadata: ' +
                str(e) +
                f" Product {url}"
            )
            metadata = 'error'

        row['metadata'] = metadata
        return row

    def download_rasters(self, row: pd.Series) -> pd.Series:

        band_resolution = f'{self.band}_{self.resolution}'

        online = row['online']
        if online != 'true':
            row[band_resolution] = []
            return row

        account = row['account']
        metadata = row['metadata']
        uuid = row['uuid']
        polygons = row['polygons']

        try:
            metadata = xmltodict.parse(metadata)
            user_product = metadata['n1:Level-2A_User_Product']
            general_info = user_product['n1:General_Info']
            product_info = general_info['Product_Info']
            product_organisation = product_info['Product_Organisation']
            granule_list = product_organisation['Granule_List']
            granule = granule_list['Granule']
            band_list = granule['IMAGE_FILE']

            band_path = next(filter(lambda x: band_resolution in x, band_list)) + '.jp2'

            product_uri = product_info['PRODUCT_URI']
            uri_band_path = f"{product_uri}/{band_path}"
            nodes_uri_band_path = map(lambda x: f"Nodes('{x}')", uri_band_path.split('/'))
            complete_band_path = '/'.join(nodes_uri_band_path)
            
            url = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/{complete_band_path}/$value"

            time.sleep(self.sleep_time)
            r = requests.get(url, auth=account, stream=True, allow_redirects=True)

            if r.status_code != 200:
                raise Exception(f'Status code {r.status_code}')

            r.raw.decode_content = True

            rasters = list()

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

                        raster = dict()
                        raster['out_image'] = out_image
                        raster['out_meta'] = out_meta

                        rasters.append(raster)

                        logging.info(f"polygon {polygon['polygon_id']} {band_resolution} successfully downloaded")
            
            row[band_resolution] = rasters

            return row
        except Exception as e:
            logging.error(
                f'Error while downloading {self.band}_{self.resolution}: ' +
                str(e) +
                f" Product {uuid}"
            )
            row[band_resolution] = []
            
            return row

    def resize_raster(self, raster: dict, scale_factor: float, resampling) -> dict:

        with MemoryFile() as memfile:
            with memfile.open(**raster['out_meta']) as src:
                src.write(raster['out_image'])

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

                resized_raster = dict()
                resized_raster['out_image'] = resized_image
                resized_raster['out_meta'] = resized_meta

                return resized_raster
    
    def resize_rasters(self, row: pd.Series) -> pd.Series:

        band_resolution = f'{self.band}_{self.resolution}'
        rasters = row[band_resolution]
        seq = map(lambda raster: self.resize_raster(raster, self.scale_factor, self.resampling), rasters)
        row[f'resized_{band_resolution}'] = list(seq)

        return row

    def save_raster(self, polygon: dict, title: str, summary: str, raster: dict) -> None:
    
        polygon_id = polygon['polygon_id']
        date = summary.split(',')[0].replace('Date: ', '').replace(':', '-')
        out_meta = raster['out_meta']
        out_image = raster['out_image']

        folders = os.path.join('rasters', polygon_id)

        if not os.path.exists(folders):
            os.makedirs(folders)

        save_path = os.path.join(folders, f'{self.band}-{date}-{title}.tif')

        with rasterio.open(save_path, 'w', **out_meta) as src:
            src.write(out_image)

    def save_rasters(self, row: pd.Series) -> None:
        
        band_resolution = f'{self.band}_{self.resolution}'
        polygons = row['polygons']
        title = row['title']
        summary = row['summary']
        rasters = row[f'resized_{band_resolution}']

        seq = map(lambda polygon,raster: self.save_raster(polygon, title, summary, raster), polygons, rasters)
        _ = list(seq)