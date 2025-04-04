# Shree KRISHNAya Namaha
# Runs all metrics serially
# Modified from AllMetrics01_N3DV.py
# Author: Nagabhushan S N
# Last Modified: 14/01/2023

import argparse
import datetime
import importlib.util
import time
import traceback
from pathlib import Path
from typing import List

import QualityScoresGrouper, QualityScoresSorter

this_filepath = Path(__file__)
this_filename = Path(__file__).stem


def run_all_specified_qa(metric_filepaths: List[Path], pred_train_dirpath: Path, database_dirpath: Path,
                         dense_model_dirpath: Path, camera_suffix: str, resolution_suffix: str, downsampling_factor: int):
    args_values = locals()
    qa_scores = {}
    for metric_file_path in metric_filepaths:
        spec = importlib.util.spec_from_file_location('module.name', metric_file_path.absolute().resolve().as_posix())
        qa_module = importlib.util.module_from_spec(spec)
        # noinspection PyUnresolvedReferences
        spec.loader.exec_module(qa_module)
        function_arguments = {}
        for arg_name in run_all_specified_qa.__code__.co_varnames[:run_all_specified_qa.__code__.co_argcount]:
            # noinspection PyUnresolvedReferences
            if arg_name in qa_module.start_qa.__code__.co_varnames[:qa_module.start_qa.__code__.co_argcount]:
                function_arguments[arg_name] = args_values[arg_name]
        # noinspection PyUnresolvedReferences
        qa_score = qa_module.start_qa(**function_arguments)
        # noinspection PyUnresolvedReferences
        qa_name = qa_module.this_metric_name
        qa_scores[qa_name] = qa_score
    return qa_scores


def run_all_qa(pred_train_dirpath: Path, database_dirpath: Path, dense_model_dirpath: Path, camera_suffix: str,
               resolution_suffix: str, downsampling_factor: int):
    frame_metric_filepaths = [
        # this_filepath.parent / '../../01_RMSE/src/RMSE02_ID.py',
        # this_filepath.parent / '../../02_PSNR/src/PSNR02_ID.py',
        # this_filepath.parent / '../../03_SSIM/src/SSIM02_ID.py',
        # this_filepath.parent / '../../04_LPIPS/src/LPIPS02_ID.py',
        this_filepath.parent / '../../05_VMAF/src/VMAF02_ID.py',

        this_filepath.parent / '../../11_DepthRMSE/src/DepthRMSE02_ID.py',
        this_filepath.parent / '../../12_DepthMAE/src/DepthMAE02_ID.py',
        this_filepath.parent / '../../13_DepthSROCC/src/DepthSROCC02_ID.py',

        this_filepath.parent / '../../05_VMAF/src/VMAF12_ID.py',
        this_filepath.parent / '../../11_DepthRMSE/src/DepthRMSE12_ID.py',
        this_filepath.parent / '../../12_DepthMAE/src/DepthMAE12_ID.py',
        this_filepath.parent / '../../13_DepthSROCC/src/DepthSROCC12_ID.py',
    ]

    qa_scores = run_all_specified_qa(frame_metric_filepaths, pred_train_dirpath, database_dirpath, dense_model_dirpath,
                                     camera_suffix, resolution_suffix, downsampling_factor)
    train_num = int(pred_train_dirpath.stem[-4:])
    QualityScoresGrouper.group_qa_scores(pred_train_dirpath.parent, [train_num])
    QualityScoresSorter.sort_qa_scores(pred_train_dirpath.parent, [train_num])
    return qa_scores


def demo1():
    pred_train_dirpath = Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1001')
    database_dirpath = Path('../../../../databases/InterDigital/data')
    gt_depth_dirpath = Path('../../../view_synthesis/literature/001_Kplanes/runs/training/train1001')
    camera_suffix = '_undistorted'
    resolution_suffix = '_down2'
    downsampling_factor = 1
    qa_scores = run_all_qa(pred_train_dirpath, database_dirpath, gt_depth_dirpath, camera_suffix, resolution_suffix, downsampling_factor)
    return qa_scores


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

    camera_suffix = args['camera_suffix']
    resolution_suffix = args['resolution_suffix']
    downsampling_factor = args['downsampling_factor']

    qa_scores = run_all_qa(pred_train_dirpath, database_dirpath, dense_model_dirpath, camera_suffix, resolution_suffix, downsampling_factor)
    return qa_scores


def demo3():
    pred_dirpaths = [
        # Path('../../../view_synthesis/literature/001_Kplanes/runs/training/train1002'),
        Path('../../../view_synthesis/literature/001_Kplanes/runs/training/train1006'),

        # Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1001'),
        # Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1002'),
        # Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1003'),
        # Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1004'),
        # Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1005'),
        # Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1006'),
        # Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1007'),
        Path('../../../view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1008'),
    ]
    for pred_train_dirpath in pred_dirpaths:
        database_dirpath = Path('../../../../databases/InterDigital/data')
        dense_model_dirpath = Path('../../../view_synthesis/literature/001_Kplanes/runs/training/train1005')
        camera_suffix = '_undistorted'
        resolution_suffix = '_down2'
        downsampling_factor = 1
        qa_scores = run_all_qa(pred_train_dirpath, database_dirpath, dense_model_dirpath, camera_suffix, resolution_suffix, downsampling_factor)
    return qa_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo3')
    parser.add_argument('--pred_train_dirpath')
    parser.add_argument('--database_dirpath')
    parser.add_argument('--dense_model_dirpath')
    parser.add_argument('--camera_suffix', default='_undistorted')
    parser.add_argument('--resolution_suffix', default='_down4')
    parser.add_argument('--downsampling_factor', type=int, default=1)
    parser.add_argument('--chat_names', nargs='+')
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'pred_train_dirpath': args.pred_train_dirpath,
        'database_dirpath': args.database_dirpath,
        'dense_model_dirpath': args.dense_model_dirpath,
        'camera_suffix': args.camera_suffix,
        'resolution_suffix': args.resolution_suffix,
        'downsampling_factor': args.downsampling_factor,
        'chat_names': args.chat_names,
    }
    return args_dict


def main(args: dict):
    if args['demo_function_name'] == 'demo1':
        qa_scores = demo1()
    elif args['demo_function_name'] == 'demo2':
        qa_scores = demo2(args)
    else:
        raise RuntimeError(f'Unknown demo function: {args["demo_function_name"]}')
    return qa_scores


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    parsed_args = parse_args()
    try:
        qa_scores_dict = main(parsed_args)
        qa_scores_str = '\n'.join([f'{key}: {value}' for key, value in qa_scores_dict.items()])
        run_result = f'Program completed successfully!\n\n{parsed_args["pred_train_dirpath"]}\n{qa_scores_str}'
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
        Telegrammer.send_message(message_content, chat_names=chat_names)
