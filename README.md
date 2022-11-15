# batchsat
This project aims to build a software that can download Sentinel-2 satellite images in batch for data analysis.

# how to install

## with *conda* (we used *miniconda3*)

1. download the project
2. create a conda environment and run the following command: `conda install rasterio=1.3.3 geopandas=0.12.1 -c conda-forge`
3. then, run the command: `pip install sentinelsat==1.1.1 xmltodict==0.13.0`

# how to use

create a csv file with the columns:

```
polygon_id;polygon_wkt;dt_from;dt_to
<unique_polygon_id_0>;<polygon_wkt_0>;<YYYY-mm-dd>;<YYYY-mm-dd>
.
.
.
<unique_polygon_id_1>;<polygon_wkt_1>;<YYYY-mm-dd>;<YYYY-mm-dd>
```

then, run `python products.py --csv-path <path_to_csv_file>`. there are other arguments, see with `python products.py -h`
you will see that a `products-polygons.csv` was created.

use it as input to the next script, running: `python rasters.py --csv-path <path_to_output_of_products.py>`. if everything goes ok, it will be created a folder named `rasters` and subfolders named with the poygons unique id. inside these subfolders you will see the downloaded raster files ready to be used.

# project goals

- build a knowledge on how to collect, prepare, process and work with raster files and raster files time series
- learn digital and image processing techniques applied to satellite images
- provide a script to concurrently download images from Sentinel-2 twin satellites using computer with relatively good resources
- provide a container/microservice to scale the download satellite images in a k8s cluster
- help programmers and researchers to easily get satellite image data to make the world a better and safer place
