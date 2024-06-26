# Shree KRISHNAya Namaha
# Estimates sparse depth on InterDigital scenes
# Modified from DepthEstimator_N3DV01.py
# Author: Nagabhushan S N
# Last Modified: 12/01/2024

import json
import time
import datetime
import traceback

import numpy
import simplejson
import skimage.io
import pandas

from pathlib import Path

import skvideo.io
from tqdm import tqdm
from matplotlib import pyplot
from deepdiff import DeepDiff

import ImagesTester03 as ImagesTester
import VideoFramesTester01 as VideoFramesTester

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_gen_num = int(this_filename[-2:])


NUM_FRAMES = 300


class Tester(VideoFramesTester.Tester):
    @staticmethod
    def read_video(path: Path):
        video = skvideo.io.vread(path.as_posix())[:NUM_FRAMES]
        return video


def start_generation(gen_configs: dict):
    root_dirpath = Path('../../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'databases' / gen_configs['database_dirpath']
    tmp_dirpath = root_dirpath / 'tmp'

    output_dirpath = database_dirpath / f"all/estimated_depths/DEL001_DE{gen_configs['gen_num']:02}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    VideoFramesTester.save_configs(output_dirpath, gen_configs)

    set_num = gen_configs['gen_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = numpy.unique(video_data['scene_name'].to_numpy())

    tester = Tester(gen_configs, database_dirpath, tmp_dirpath, ImagesTester.ColmapTester)

    failed_frames_path = output_dirpath / 'FailedFrameNums.csv'
    if failed_frames_path.exists():
        failed_pairs_data = pandas.read_csv(failed_frames_path)
    else:
        failed_pairs_data = pandas.DataFrame(columns=['scene_name', 'frame_num'])

    for scene_name in scene_names:
        # if scene_name not in ['coffee_martini']:
        #     continue

        tester.setup(scene_name)
        video_nums = video_data.loc[video_data['scene_name'] == scene_name]['pred_video_num'].to_numpy()
        num_frames = NUM_FRAMES

        for frame_num in tqdm(range(num_frames), desc=scene_name):
            bounds_path = output_dirpath / f'{scene_name}/estimated_bounds/{frame_num:04}.csv'
            is_failed_pair = failed_pairs_data.loc[
                                 (failed_pairs_data['scene_name'] == scene_name) &
                                 (failed_pairs_data['frame_num'] == frame_num)
                             ].shape[0] > 0
            if bounds_path.exists() or is_failed_pair:
                continue
            depth_data_list, bounds_data = tester.get_sparse_depth(frame_num)
            if (depth_data_list is not None) and (len(depth_data_list) == video_nums.size):
                for i, video_num in enumerate(video_nums):
                    depth_path = output_dirpath / f'{scene_name}/estimated_depths/{video_num:04}_{frame_num:04}.csv'
                    depth_path.parent.mkdir(parents=True, exist_ok=True)
                    depth_data_list[i].to_csv(depth_path, index=False)
                bounds_path.parent.mkdir(parents=True, exist_ok=True)
                bounds_data.to_csv(bounds_path, index=False)
            else:
                frame_data = pandas.DataFrame.from_dict({
                    'scene_name': [scene_name],
                    'frame_num': [frame_num],
                })
                failed_pairs_data = pandas.concat([failed_pairs_data, frame_data], ignore_index=True)
                failed_pairs_data.to_csv(failed_frames_path, index=False)
    return


def demo1():
    """
    For a gen set
    :return:
    """
    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 4,
        'gen_set_num': 4,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 5,
        'gen_set_num': 5,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 3,
        'gen_set_num': 3,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 1,
        'gen_set_num': 1,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 11,
        'gen_set_num': 11,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 12,
        'gen_set_num': 12,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 13,
        'gen_set_num': 13,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 14,
        'gen_set_num': 14,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': f'{this_filename}/{VideoFramesTester.this_filename}/{ImagesTester.this_filename}',
        'gen_num': 15,
        'gen_set_num': 15,
        'database_name': 'InterDigital',
        'database_dirpath': 'InterDigital/data',
        'camera_suffix': '_undistorted',
        'resolution_suffix': '_down2',
    }
    start_generation(gen_configs)
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
