# Shree KRISHNAya Namaha
# Reconciles and saves failed frames data for a folder that has already been generated
# Modified from FEL001/FailedPairsReconciler.py
# Author: Nagabhushan S N
# Last Modified: 02/01/2024

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


def reconcile_failed_frames_data(database_dirpath: Path, depth_dirpath: Path, num_frames: int):
    configs_path = depth_dirpath / 'configs.json'
    with open(configs_path, 'r') as configs_file:
        configs = simplejson.load(configs_file)
    set_num = configs['gen_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = numpy.unique(video_data['scene_name'].to_numpy())

    failed_frames_path = depth_dirpath / 'FailedFrames.csv'
    if not failed_frames_path.exists():
        failed_frames_data = pandas.read_csv(failed_frames_path)
    else:
        failed_frames_data = pandas.DataFrame(columns=['scene_name', 'frame_num'])

    new_failed_frames_list = []
    for scene_name in scene_names:
        video_nums = video_data[video_data['scene_name'] == scene_name]['pred_video_num'].to_numpy()

        for frame_num in range(num_frames):
            output_path = depth_dirpath / f'{scene_name}/estimated_bounds/{frame_num:04}.csv'
            is_failed_pair = failed_frames_data.loc[
                                 (failed_frames_data['scene_name'] == scene_name) &
                                 (failed_frames_data['frame_num'] == frame_num)
                             ].shape[0] > 0
            if not (output_path.exists() or is_failed_pair):
                new_failed_frames_list.append([scene_name, frame_num])

    new_failed_frames_data = pandas.DataFrame(new_failed_frames_list, columns=['scene_name', 'frame_num'])
    failed_frames_data = pandas.concat([failed_frames_data, new_failed_frames_data], ignore_index=True)
    failed_frames_data.to_csv(failed_frames_path, index=False)
    return


def demo1():
    root_dirpath = Path('../../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'databases/N3DV'
    depth_dirpath = database_dirpath / 'all/estimated_depths/DEL001_DE04'

    reconcile_failed_frames_data(database_dirpath, depth_dirpath, num_frames=300)
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
