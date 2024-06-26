# Shree KRISHNAya Namaha
# Combines all the depth bounds files into a single file
# Modified from DepthAggregator.py
# Author: Nagabhushan S N
# Last Modified: 15/01/2024

import time
import datetime
import traceback
import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def aggregate_bounds_data(database_dirpath_suffix: str, depth_dirname: str):
    root_dirpath = Path('../../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'databases' / database_dirpath_suffix
    depth_dirpath = database_dirpath / f'estimated_depths/{depth_dirname}'
    for scene_dirpath in sorted(depth_dirpath.iterdir()):
        if not scene_dirpath.is_dir():
            continue
        all_bounds_arrays = []
        for bounds_path in tqdm(sorted((scene_dirpath / 'estimated_bounds').glob('*.csv')), desc=scene_dirpath.stem):
            bounds_data = pandas.read_csv(bounds_path)
            all_bounds_arrays.append(bounds_data.to_numpy())
        all_bounds_arrays = numpy.stack(all_bounds_arrays, axis=0)  # (num_frames,  num_views, 2)
        near_array = numpy.min(all_bounds_arrays[:, :, 0], axis=0)  # (num_views,)
        far_array = numpy.max(all_bounds_arrays[:, :, 1], axis=0)  # (num_views,)
        bounds_array = numpy.stack([near_array, far_array], axis=1)  # (num_views, 2)
        bounds_data = pandas.DataFrame(bounds_array, columns=['near', 'far'])
        output_path = scene_dirpath / 'EstimatedBounds.csv'
        bounds_data.to_csv(output_path, index=False)
    return


def demo1():
    database_dirpath_suffix = 'InterDigital/data/all'
    depth_dirname = 'DEL001_DE01'
    aggregate_bounds_data(database_dirpath_suffix, depth_dirname)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
