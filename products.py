import time
import json
import logging
import argparse
import requests
import pandas as pd
import shapely.wkt as wkt

from datetime import date
from functools import reduce
from itertools import repeat
from sentinelsat import SentinelAPI
from concurrent.futures import ThreadPoolExecutor


def products_from_polygons(
    account: tuple,
    polygon_id: str, polygon_wkt: str, dt_from: date, dt_to: date,
    cloudcoverpercentage: tuple = (0, 100),
    platformname: str = 'Sentinel-2', producttype: str = 'S2MSI2A',
    processinglevel = 'Level-2A'
) -> pd.DataFrame | None:
    
    api = SentinelAPI(account[0], account[1], 'https://apihub.copernicus.eu/apihub')
    products: pd.DataFrame | None = None
    
    try:
        polygon_envelope = wkt.loads(polygon_wkt).envelope.wkt
        
        products = api.query(
            polygon_envelope,
            date=(dt_from.strftime("%Y%m%d"), dt_to.strftime("%Y%m%d")),
            platformname=platformname,
            producttype=producttype,
            processinglevel=processinglevel,
            cloudcoverpercentage=cloudcoverpercentage
        )
        
        products = api.to_dataframe(products)
        
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
    
    return products

def product_online_status(account: tuple, uuid: str) -> str:

    url: str = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/Online/$value"
    status: str | None = None

    try:
        time.sleep(0.125)

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
    
    return status

def trigger_online(account: tuple, uuid: str, online: str) -> str:

    if online == 'true':
        return 'false'

    url: str = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/$value"
    status: str | None = None

    try:
        time.sleep(0.125)
        r = requests.get(url, auth=account, allow_redirects=True)

        if r.status_code != 202:
            raise Exception(f'Status code {r.status_code}')

        status = 'true'
        logging.info(f"Product '{uuid}' successfully triggered to online, try to download in a few hours")
    except Exception as e:
        logging.error(
            'Error while triggering offline product: ' +
            str(e) +
            f" Product {url}"
        )
        status = 'error'
    
    return status

def product_metadata(account: tuple, uuid: str, title: str, online: str) -> str:

    if online != 'true':
        return 'error'

    url: str = f"https://apihub.copernicus.eu/apihub/odata/v1/Products('{uuid}')/Nodes('{title}.SAFE')/Nodes('MTD_MSIL2A.xml')/$value"
    metadata: str | None = None

    try:
        time.sleep(0.125)
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

    return metadata

def repeat_list(_list: list, size: int) -> list:

    new_list = []

    while len(new_list) < size:
        new_list += _list

    return new_list[:size]

def main(
    csv_path: str, samples: int,
    random_state: int, max_workers: int,
    cloudcoverpercentage: tuple
) -> None:

    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(
        csv_path, sep=';', low_memory = False,
        dtype = str,
        parse_dates=['dt_from', 'dt_to']
    )

    if samples is not None:
        df = df.sample(samples, random_state = random_state)

    accs = [('copernicus_user', 'copernicus_password')]
    accounts = repeat_list(accs, len(df))

    concat_dfs = lambda x, y: pd.concat([x, y], ignore_index=True)
    is_polygon_inside = lambda x: wkt.loads(x['footprint']).contains(wkt.loads(x['polygon']['polygon_envelope']))
    is_not_none = lambda x: x is not None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # get all the products and filter products that not fully contain its polygon
        seq = executor.map(products_from_polygons, accounts, df.polygon_id, df.polygon_wkt, df.dt_from, df.dt_to, repeat(cloudcoverpercentage))
        seq = filter(is_not_none, seq)
        df = reduce(concat_dfs, seq)
        df = df[df.apply(is_polygon_inside, axis=1)]
        size_before = df.shape[0]

        # group polygons by products, they might have repeated polygons per product
        df  = df.groupby(
            ['uuid', 'title', 'link', 'link_alternative', 'summary', 'footprint', 'filename', 'identifier', 'ondemand'],
            as_index=False
        ).agg({'polygon': lambda x: list(x)})
        size_after = df.shape[0]

        logging.info(
            f"Processing {size_after} products. " +
            f"Reusing products that contains one or more polygons (Reduction: {round(100*(1. - size_after/size_before),2)}%)"
        )

        df = df.rename(columns={'polygon': 'polygons'})
        df['polygons'] = df['polygons'].apply(json.dumps)

        # check if all the products is online and ready to download
        accounts = repeat_list(accs, len(df))
        seq = executor.map(product_online_status, accounts, df.uuid)
        df['online'] = list(seq)

        # trigger the online status from offline products
        seq = executor.map(trigger_online, accounts, df.uuid, df.online)
        df['triggered'] = list(seq)

        # get metadata from online products
        seq = executor.map(product_metadata, accounts, df.uuid, df.title, df.online)
        df['metadata'] = list(seq)

        df = df.query('online == \'true\' and metadata != \'error\'')

        df.to_csv('products-polygons.csv', sep=';', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Search and retrieve products based on polygon polygons and temporal window.')

    parser.add_argument('--csv-path',
        help='file path for the csv with polygons id, wkt and temporal window',
        required=True
    )

    parser.add_argument('--samples',
        type=int, default=None,
        help='an integer for the number of samples (default: None, i.e. use all)'
    )

    parser.add_argument('--random-state',
        type=int, default=42,
        help='an integer for the random state'
    )

    parser.add_argument('--max-workers',
        type=int, default=5,
        help='an integer for the number of concurrent threads to execute Copernicus api calls (default: 5)'
    )

    parser.add_argument('--cloudcoverpercentage',
        type=str, default='0-50',
        help='a tuple of integers for the range of cloud cover percentage to be (default: \'0-50\')'
    )

    args = parser.parse_args()
    args = vars(args)
    args['cloudcoverpercentage'] = tuple(args['cloudcoverpercentage'].split('-'))

    main(**args)