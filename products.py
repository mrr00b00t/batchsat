import time
import json
import logging
import argparse
import requests
import pandas as pd
import shapely.wkt as wkt
from functools import reduce
from itertools import repeat
from datetime import date
from sentinelsat import SentinelAPI
from concurrent.futures import ThreadPoolExecutor

from random import choice


def products_from_crops(
    account: tuple,
    crop_id: str, crop_wkt: str, dt_from: date, dt_to: date,
    cloudcoverpercentage: tuple = (0, 100),
    platformname: str = 'Sentinel-2', producttype: str = 'S2MSI2A'
) -> pd.DataFrame | None:
    
    api = SentinelAPI(account[0], account[1], 'https://apihub.copernicus.eu/apihub')
    products: pd.DataFrame | None = None
    
    try:
        crop_envelope = wkt.loads(crop_wkt).envelope.wkt
        
        products = api.query(
            crop_envelope,
            date=(dt_from.strftime("%Y%m%d"), dt_to.strftime("%Y%m%d")),
            platformname=platformname,
            producttype=producttype,
            cloudcoverpercentage=cloudcoverpercentage
        )
        
        products = api.to_dataframe(products)
        
        products['crop'] = products['uuid'].apply(lambda _ : {
            'crop_id': crop_id,
            'crop_wkt': crop_wkt,
            'crop_envelope': crop_envelope
        })
        logging.info(f"Retrieved {len(products)} products from crop {crop_id}")
    except Exception as e:
        logging.error(
            'Error while retrieving products: ' +
            str(e) +
            f" Crop {crop_id}"
        )
    
    return products

def product_online_status(account: tuple, uuid: str) -> str:

    url: str = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/Online/$value"
    status: str | None = None

    try:
        time.sleep(0.125)
        status = requests.get(url, auth=account).content.decode('utf-8')
        logging.info(f"{uuid} is online? -> {status}")
    except Exception as e:
        logging.error(
            'Error while checking online: ' +
            str(e) +
            f" Product {uuid}"
        )
        status = 'error'
    
    return status

def trigger_online(account: tuple, uuid: str, online: str) -> str:

    if online == 'true':
        return 'false'

    url: str = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/$value"
    status: str | None = None

    try:
        time.sleep(0.125)
        _ = requests.get(url, auth=account, allow_redirects=True)
        status = 'true'
        logging.info(f"Product '{uuid}' successfully triggered to online, try to download in a few hours")
    except Exception as e:
        logging.error(
            'Error while triggering offline product: ' +
            str(e) +
            f" Product {uuid}"
        )
        status = 'error'
    
    return status

def product_metadata(account: tuple, uuid: str, title: str, online: str) -> str:

    if online == 'false':
        return 'error'

    url: str = f"https://apihub.copernicus.eu/apihub/odata/v1/Products('{uuid}')/Nodes('{title}.SAFE')/Nodes('MTD_MSIL2A.xml')/$value"
    metadata: str | None = None

    try:
        time.sleep(0.125)
        metadata = requests.get(url, auth=account).content.decode('utf-8')
        logging.info(f"Product {uuid} metadata successfully retrieved")
    except Exception as e:
        logging.error(
            'Error while retrieving metadata: ' +
            str(e) +
            f" Product {uuid}"
        )
        metadata = 'error'

    return metadata

def repeat_list(_list: list, size: int) -> list:

    new_list = []

    while len(new_list) < size:
        new_list += _list

    return new_list[:size]

def main(
    csv_path: str,
    random_state: int = 42, samples: int = 30,
    max_workers: int = 10, cloudcoverpercentage: tuple = (0, 30)
) -> None:

    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(
        csv_path, sep=';', low_memory = False,
        dtype = str,
        parse_dates=['dt_from', 'dt_to']
    ).sample(samples, random_state = random_state)

    accs = [('username', 'password')]
    accounts = repeat_list(accs, len(df))

    concat_dfs = lambda x, y: pd.concat([x, y], ignore_index=True)
    is_crop_inside = lambda x: wkt.loads(x['footprint']).contains(wkt.loads(x['crop']['crop_envelope']))
    is_not_none = lambda x: x is not None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # get all the products and filter products that not fully contain its crop
        seq = executor.map(products_from_crops, accounts, df.crop_id, df.crop_wkt, df.dt_from, df.dt_to, repeat(cloudcoverpercentage))
        df = filter(is_not_none, df)
        df = reduce(concat_dfs, seq)
        df = df[df.apply(is_crop_inside, axis=1)]
        size_before = df.shape[0]

        # group crops by products, they might have repeated crops per product
        df  = df.groupby(
            ['uuid', 'title', 'link', 'link_alternative', 'summary', 'footprint', 'filename', 'identifier'],
            as_index=False
        ).agg({'crop': lambda x: list(x)})
        size_after = df.shape[0]

        logging.info(
            f"Processing {size_after} products. " +
            f"Reusing products that contains one or more crops (Reduction rate: {1. - size_after/size_before})"
        )

        df = df.rename(columns={'crop': 'crops'})
        df['crops'] = df['crops'].apply(json.dumps)

        # check if all the products is online and ready to download
        accounts = repeat_list(accs, len(df))
        df['online'] = [r for r in executor.map(product_online_status, accounts, df.uuid)]

        # trigger the online status from offline products
        df['triggered'] = [r for r in executor.map(trigger_online, accounts, df.uuid, df.online)]

        # get metadata from online products
        df['metadata'] = [r for r in executor.map(product_metadata, accounts, df.uuid, df.title, df.online)]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Search and retrieve products based on crop polygons and temporal window.')

    parser.add_argument('--csv-path',
        help='file path for the csv with crops id, wkt and temporal window',
        required=True
    )

    parser.add_argument('--random-state',
        type=int,
        help='an integer for the random state'
    )

    parser.add_argument('--samples',
        type=int, default=30,
        help='an integer for the number of samples (default: 30, i.e. use all)'
    )

    parser.add_argument('--max-workers',
        type=int, default=10,
        help='an integer for the number of concurrent threads to execute Copernicus api calls (default: 10)'
    )

    parser.add_argument('--cloudcoverpercentage',
        type=str, default='0-50',
        help='a tuple of integers for the range of cloud cover percentage to be (default: \'0-50\')'
    )

    args = parser.parse_args()
    args = vars(args)
    args['cloudcoverpercentage'] = tuple(args['cloudcoverpercentage'].split('-'))

    main(**args)