# Shree KRISHNAya Namaha
# VMAF measure between predicted spiral video and spiral video from dense input NeRF
# Modified from VMAF11_N3DV.py
# Author: Nagabhushan S N
# Last Modified: 14/01/2024

import argparse
import datetime
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
import skvideo.io
from tqdm import tqdm

from vmaf.script.run_vmaf import python_hook as vmaf_hook

import CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_metric_name = this_filename[:-3]

NUM_FRAMES = 300
RESOLUTION = (544, 1024)


class VMAF:
    def __init__(self, videos_data: pandas.DataFrame, tmp_dirpath: Path = None, verbose_log: bool = True) -> None:
        super().__init__()
        self.videos_data = videos_data
        self.verbose_log = verbose_log
        self.tmp_dirpath = self.get_tmp_dirpath(tmp_dirpath)
        return

    @staticmethod
    def get_tmp_dirpath(tmp_dirpath: Path):
        if tmp_dirpath is None:
            tmp_dirpath = this_filepath.parent.parent / f'tmp/{this_filename}'
        if tmp_dirpath.exists():
            shutil.rmtree(tmp_dirpath)
        tmp_dirpath.mkdir(parents=True, exist_ok=False)
        return tmp_dirpath

    def compute_vmaf(self, gt_rgb_path: Path, eval_rgb_path: Path):
        gt_rgb_yuv_path = self.tmp_dirpath / 'gt_rgb.yuv'
        eval_rgb_yuv_path = self.tmp_dirpath / 'eval_rgb.yuv'

        cmd = f'ffmpeg -y -i "{gt_rgb_path}" -pix_fmt yuv420p "{gt_rgb_yuv_path}"'
        os.system(cmd)
        cmd = f'ffmpeg -y -i "{eval_rgb_path}" -pix_fmt yuv420p "{eval_rgb_yuv_path}"'
        os.system(cmd)

        vmaf_str = vmaf_hook(eval_rgb_yuv_path.as_posix(), gt_rgb_yuv_path.as_posix(), fmt='yuv420p', width=RESOLUTION[1], height=RESOLUTION[0], out_fmt='json')
        vmaf_dict = json.loads(vmaf_str)

        # Extract frame-wise vmaf scores into a list
        vmaf_scores_list = [frame['VMAF_score'] for frame in vmaf_dict['frames']]

        # Extract the aggregate vmaf score
        vmaf_score = vmaf_dict['aggregate']['VMAF_score']
        assert numpy.allclose(numpy.mean(vmaf_scores_list), vmaf_score)

        return vmaf_scores_list

    def compute_avg_vmaf(self, old_data: pandas.DataFrame, dense_model_dirpath: Path, pred_train_dirpath: Path,
                         iter_num: int, resolution_suffix: str, downsampling_factor: int):
        """

        :param old_data:
        :param dense_model_dirpath:
        :param pred_train_dirpath:
        :param iter_num:
        :param resolution_suffix:
        :param downsampling_factor:
        :return:
        """
        qa_scores = []
        scene_names = self.videos_data['scene_name'].unique()
        for scene_name in tqdm(scene_names, leave=self.verbose_log):
            pred_video_num = 0

            if old_data is not None and old_data.loc[
                (old_data['scene_name'] == scene_name) &
                (old_data['pred_video_num'] == pred_video_num) &
                (old_data['pred_frame_num'] == NUM_FRAMES-1)
            ].size > 0:
                continue

            gt_rgb_path = dense_model_dirpath / f'{scene_name}/predicted_videos_iter{90000:06}/rgb/{pred_video_num:04}_spiral01.mp4'
            pred_rgb_path = pred_train_dirpath / f'{scene_name}/predicted_videos_iter{iter_num:06}/rgb/{pred_video_num:04}_spiral01.mp4'
            if not (gt_rgb_path.exists() and pred_rgb_path.exists()):
                continue
            if downsampling_factor > 1:
                raise NotImplementedError

            vmaf_scores = self.compute_vmaf(gt_rgb_path, pred_rgb_path)
            for frame_num, vmaf_score in enumerate(vmaf_scores):
                qa_scores.append([scene_name, pred_video_num, frame_num, vmaf_score])
        qa_scores_data = pandas.DataFrame(qa_scores, columns=['scene_name', 'pred_video_num', 'pred_frame_num', this_metric_name])

        merged_data = self.update_qa_frame_data(old_data, qa_scores_data)
        avg_vmaf = numpy.mean(merged_data[this_metric_name])
        merged_data = merged_data.round({this_metric_name: 4, })
        avg_vmaf = numpy.round(avg_vmaf, 4)
        if isinstance(avg_vmaf, numpy.floating):
            avg_vmaf = avg_vmaf.item()
        return avg_vmaf, merged_data

    @staticmethod
    def update_qa_frame_data(old_data: pandas.DataFrame, new_data: pandas.DataFrame):
        if (old_data is not None) and (new_data.size > 0):
            old_data = old_data.copy()
            new_data = new_data.copy()
            old_data.set_index(['scene_name', 'pred_video_num', 'pred_frame_num'], inplace=True)
            new_data.set_index(['scene_name', 'pred_video_num', 'pred_frame_num'], inplace=True)
            merged_data = old_data.combine_first(new_data).reset_index()
        elif old_data is not None:
            merged_data = old_data
        else:
            merged_data = new_data
        return merged_data

    @classmethod
    def read_video(cls, video_path: Path):
        video = skvideo.io.vread(video_path.as_posix())
        return video

    @staticmethod
    def downsample_video(video: numpy.ndarray, downsampling_factor: int):
        downsampled_video = skvideo.transform.rescale(video, scale=1 / downsampling_factor, preserve_range=True,
                                                      multichannel=False, anti_aliasing=True)
        return downsampled_video


def get_iter_nums(pred_train_dirpath: Path):
    iter_nums = []
    for pred_videos_dirpath in sorted(pred_train_dirpath.glob('**/predicted_videos_iter*')):
        iter_num = int(pred_videos_dirpath.stem[-6:])
        iter_nums.append(iter_num)
    iter_nums = numpy.unique(iter_nums).tolist()
    return iter_nums


# noinspection PyUnusedLocal
def start_qa(pred_train_dirpath: Path, database_dirpath: Path, dense_model_dirpath: Path, resolution_suffix,
             downsampling_factor: int):
    if not pred_train_dirpath.exists():
        print(f'Skipping QA of folder: {pred_train_dirpath.stem}. Reason: pred_train_dirpath does not exist')
        return

    if not dense_model_dirpath.exists():
        print(f'Skipping QA of folder: {pred_train_dirpath.stem}. Reason: dense_model_dirpath does not exist')
        return

    test_videos_datapath = CommonUtils.get_test_videos_datapath(database_dirpath, pred_train_dirpath)
    videos_data = pandas.read_csv(test_videos_datapath)[['scene_name', 'pred_video_num']]
    vmaf_computer = VMAF(videos_data)

    qa_scores_filepath = pred_train_dirpath / 'QualityScores.json'
    iter_nums = get_iter_nums(pred_train_dirpath)
    avg_scores = {}
    for iter_num in iter_nums:
        if qa_scores_filepath.exists():
            with open(qa_scores_filepath.as_posix(), 'r') as qa_scores_file:
                qa_scores = json.load(qa_scores_file)
        else:
            qa_scores = {}

        if str(iter_num) in qa_scores:
            if this_metric_name in qa_scores[str(iter_num)]:
                avg_vmaf = qa_scores[str(iter_num)][this_metric_name]
                print(f'Average {this_metric_name}: {pred_train_dirpath.as_posix()} - {iter_num:06}: {avg_vmaf}')
                print('Running QA again.')
        else:
            qa_scores[str(iter_num)] = {}

        vmaf_data_path = pred_train_dirpath / f'quality_scores/iter{iter_num:06}/{this_metric_name}_FrameWise.csv'
        if vmaf_data_path.exists():
            vmaf_data = pandas.read_csv(vmaf_data_path)
        else:
            vmaf_data = None

        avg_vmaf, vmaf_data = vmaf_computer.compute_avg_vmaf(vmaf_data, dense_model_dirpath, pred_train_dirpath, iter_num,
                                                             resolution_suffix, downsampling_factor)
        if numpy.isfinite(avg_vmaf):
            avg_scores[iter_num] = avg_vmaf
            qa_scores[str(iter_num)][this_metric_name] = avg_vmaf
            print(f'Average {this_metric_name}: {pred_train_dirpath.as_posix()} - {iter_num:06}: {avg_vmaf}')
            with open(qa_scores_filepath.as_posix(), 'w') as qa_scores_file:
                simplejson.dump(qa_scores, qa_scores_file, indent=4)
            vmaf_data_path.parent.mkdir(parents=True, exist_ok=True)
            vmaf_data.to_csv(vmaf_data_path, index=False)
    return avg_scores


def demo1():
    pred_train_dirpath = Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1001')
    database_dirpath = Path('../../../../databases/InterDigital/data')
    dense_model_dirpath = Path('../../../view_synthesis/literature/001_Kplanes/runs/training/train1001')
    resolution_suffix = '_down2'
    downsampling_factor = 1
    avg_score = start_qa(pred_train_dirpath, database_dirpath, dense_model_dirpath, resolution_suffix, downsampling_factor)
    return avg_score


def demo2(args: dict):
    pred_train_dirpath = args['pred_train_dirpath']
    if pred_train_dirpath is None:
        raise RuntimeError(f'Please provide pred_train_dirpath')
    pred_train_dirpath = Path(pred_train_dirpath)

    database_dirpath = args['database_dirpath']
    if database_dirpath is None:
        raise RuntimeError(f'Please provide database_dirpath')
    database_dirpath = Path(database_dirpath)

    dense_model_dirpath = args['dense_model_dirpath']
    if dense_model_dirpath is None:
        raise RuntimeError(f'Please provide dense_model_dirpath')
    dense_model_dirpath = Path(dense_model_dirpath)

    resolution_suffix = args['resolution_suffix']
    downsampling_factor = args['downsampling_factor']

    avg_score = start_qa(pred_train_dirpath, database_dirpath, dense_model_dirpath, resolution_suffix, downsampling_factor)
    return avg_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--pred_train_dirpath')
    parser.add_argument('--database_dirpath')
    parser.add_argument('--dense_model_dirpath')
    parser.add_argument('--resolution_suffix', default='_down4')
    parser.add_argument('--downsampling_factor', type=int, default=1)
    parser.add_argument('--chat_names', nargs='+')
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'pred_train_dirpath': args.pred_train_dirpath,
        'database_dirpath': args.database_dirpath,
        'dense_model_dirpath': args.dense_model_dirpath,
        'resolution_suffix': args.resolution_suffix,
        'downsampling_factor': args.downsampling_factor,
        'chat_names': args.chat_names,
    }
    return args_dict


def main(args: dict):
    if args['demo_function_name'] == 'demo1':
        avg_score = demo1()
    elif args['demo_function_name'] == 'demo2':
        avg_score = demo2(args)
    else:
        raise RuntimeError(f'Unknown demo function: {args["demo_function_name"]}')
    return avg_score


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    parsed_args = parse_args()
    try:
        output_score = main(parsed_args)
        run_result = f'Program completed successfully!\nAverage {this_metric_name}: {output_score}'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = "Error: " + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

    if parsed_args['chat_names'] is not None:
        from snb_utils import Telegrammer

        chat_names = parsed_args['chat_names']
        message_content = f'QA/{this_filename} has finished.\n' + run_result
        Telegrammer.send_message(message_content, chat_names)
