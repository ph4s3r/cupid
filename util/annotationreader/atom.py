
import os
if os.name == 'nt':
    import helpers.openslideimport #on windows, openslide needs to be installed manually, check local openslideimport.py

import geojson
import numpy as np
import matplotlib.path as mp
from shapely.geometry import shape

def annotationToMask(slide_shape: tuple(), slide_name, geojson_file_path=str) -> np.ndarray:
    # converts geojson file of H&E image into pathml slide mask array object
    
    # read & load with geojson lib
    try:
        with open(str(geojson_file_path), "r") as file:
            data = geojson.load(file)
    except FileNotFoundError:
        print(f"No annotation file found for {slide_name}.")
        raise
    except Exception as err:
        print(f"Could not read annotation file {geojson_file_path}: {err}")
        raise

    geojson_struct = None
    shapely_polygons = list()

    # validate geojson data & convert to shapely polygons
    if data[0]["geometry"]["type"] == "Polygon":
        geojson_struct = geojson.Polygon(geometry=data[0]["geometry"])
        shapely_polygons.append(shape(data[0]["geometry"]))
    elif data[0]["geometry"]["type"] == "MultiPolygon":
        print(
            f"DEBUG: Multipolygon has {len(data[0]['geometry']['coordinates'])} number of polygons"
        )
        geojson_struct = geojson.MultiPolygon(geometry=data[0]["geometry"])
        data[0]["geometry"]["type"] = "Polygon"  # changing to polygon since we break it up. Multipolygon is precessed differently by shapely
        for poly in data[0]["geometry"]["coordinates"]:
            newpoly = {"type": "Polygon", "coordinates": poly}
            shapely_polygons.append(shape(newpoly))
    else:
        print(
            f"ERROR: Unexpected GeoJson object with type {data[0]['geometry']['type'] } in {slide_name}"
        )

    if not geojson_struct.is_valid:
        print(f"{geojson_file_path} has invalid GeoJson structure, skipping this.")
        print(f"errors: {geojson_struct.errors()}")
        raise

    del geojson_struct

    masks = []
    print(f"processing {len(shapely_polygons)} polygon(s) for {slide_name}")

    for i, polygon in enumerate(shapely_polygons):
        minx, miny, maxx, maxy = polygon.bounds
        minx, miny, maxx, maxy = (
            int(np.ceil(minx)),
            int(np.ceil(miny)),
            int(np.ceil(maxx)),
            int(np.ceil(maxy)),
        )
        width = maxx - minx
        height = maxy - miny

        # convert the polygon to a Path for matplotlib
        poly_path = mp.Path(polygon.exterior.coords)

        # create a grid of the same size as the bounding box
        meshgrid_x = np.arange(minx, maxx)
        meshgrid_y = np.arange(miny, maxy)

        x, y = np.meshgrid(meshgrid_x, meshgrid_y)
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        print(f"passing {len(points)} points to contains_points")

        # use the path to determine if points are inside the polygon
        grid = poly_path.contains_points(points)
        grid = grid.reshape(height, width)
        masks.append(grid)
        print(f"mask of size {grid.shape} created for polygon {i+1}")

    # initialize the mask
    mask_array = np.zeros(slide_shape, dtype=np.uint8)
    
    # add up and place masks into their correct position in the overall mask_array
    for grid, (minx, miny, maxx, maxy) in zip(
        masks,
        [
            (
                int(np.ceil(polygon.bounds[0])),
                int(np.ceil(polygon.bounds[1])),
                int(np.ceil(polygon.bounds[2])),
                int(np.ceil(polygon.bounds[3])),
            )
            for polygon in shapely_polygons
        ],
    ):
        height, width = grid.shape

        # ensure the indexing does not exceed the dimensions of mask_array
        endx, endy = min(mask_array.shape[1], minx + width), min(
            mask_array.shape[0], miny + height
        )

        # combine using binary 'or' operation
        mask_array[miny:endy, minx:endx] |= grid[: endy - miny, : endx - minx]

    return mask_array
