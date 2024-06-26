# Shree KRISHNAya Namaha
# Combines all the sparse depth files into a single file
# Modified from FEL001/FlowAggregator.py
# Author: Nagabhushan S N
# Last Modified: 02/01/24

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


def aggregate_depth_data(depth_data: pandas.DataFrame, new_depth_data: pandas.DataFrame):
    if depth_data is None:
        depth_data = new_depth_data
    else:
        depth_data = pandas.concat([depth_data, new_depth_data], axis=0)
    return depth_data


def aggregate_depth_priors(database_dirpath_suffix: str, depth_dirname: str):
    root_dirpath = Path('../../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'databases' / database_dirpath_suffix
    depth_dirpath = database_dirpath / f'estimated_depths/{depth_dirname}'
    for scene_dirpath in sorted(depth_dirpath.iterdir()):
        if not scene_dirpath.is_dir():
            continue
        all_depth_data = []
        for depth_path in tqdm(sorted((scene_dirpath / 'estimated_depths').glob('*.csv')), desc=scene_dirpath.stem):
            depth_data = pandas.read_csv(depth_path)
            column_names = list(depth_data.columns)
            video_num, frame_num = depth_path.stem.split('_')
            depth_data['video_num'] = int(video_num)
            depth_data['frame_num'] = int(frame_num)
            depth_data = depth_data[['video_num', 'frame_num'] + column_names]
            all_depth_data.append(depth_data)
        all_depth_data = pandas.concat(all_depth_data, axis=0)
        output_path = scene_dirpath / 'EstimatedDepths.csv'
        all_depth_data.to_csv(output_path, index=False)
    return


def demo1():
    database_dirpath_suffix = 'N3DV/data/all'
    depth_dirname = 'DEL001_DE20'
    aggregate_depth_priors(database_dirpath_suffix, depth_dirname)
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
