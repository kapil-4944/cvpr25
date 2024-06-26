### Author : Kapil Choudhary
### Last Modified : june 3, 2024
### Description : This script is used to find the frames with highest QA metric difference between two models



import argparse
import os
import time
import datetime
import traceback
import pandas as pd

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def max_QA_diff_frames(configs):
    scene_dirpaths1 = sorted(configs['train_dirpath1'].iterdir())
    scene_dirpaths2 = sorted(configs['train_dirpath2'].iterdir())
    df = pd.DataFrame()
    for scene_dirpath1, scene_dirpath2 in zip(scene_dirpaths1, scene_dirpaths2):
        if  scene_dirpath1.is_dir() and scene_dirpath2.is_dir():
            scene_1 = scene_dirpath1.stem
            scene_2 = scene_dirpath2.stem
            if scene_1 == scene_2:
            # for qa_metric_dirpath1 in sorted(scene_dirpath1.glob(f'quality_scores_iter*{configs["iter_num1"]}')):
                qa_metric_dirpath1 = scene_dirpath1/ f'quality_scores_iter{configs["iter_num1"]:06d}'
                qa_metric_dirpath2 = scene_dirpath2/ f'quality_scores_iter{configs["iter_num2"]:06d}'
                if qa_metric_dirpath1.is_dir() and qa_metric_dirpath2.is_dir():
                    for qa_name in configs['qa_metric_names']:
                        qa_dirpath1  = qa_metric_dirpath1 / f'{qa_name}.csv'
                        qa_data1 = pd.read_csv(qa_dirpath1)
                        qa_dirpath2 = qa_metric_dirpath2 / f'{qa_name}.csv'
                        qa_data2 = pd.read_csv(qa_dirpath2)
                        # iterate over the frames and find the frames with highest QA metric difference
                        if  (qa_data1['video_num'] == qa_data2['video_num']).all():
                            QA_diff = qa_data1[qa_name] - qa_data2[qa_name]
                        # find the frames with highest QA metric difference
                            max_diff_frame = QA_diff.idxmax()
                        # index of the top 5 frames with highest QA metric difference
                        #     top5_diff_frames_idx = QA_diff.nlargest(5).index
                        # find the top 5 frames with highest QA metric difference with index and save them
                            top5_diff_frames = QA_diff.nlargest(5)
                        # find least 5 frames with highest QA metric difference and save them
                            least5_diff_frames = QA_diff.nsmallest(5)
                            QA_name = qa_name
                            scene_name = scene_1
                            dict = {'scene_name':scene_name,'QA_name': QA_name , 'top5_diff_frame_idx': top5_diff_frames.index ,'least5_diff_frame_idx': least5_diff_frames.index,  'video_num': qa_data1['video_num'][0]}
                        # make a dataframe and save dict according to the scene and QA metric
                            df1 = pd.DataFrame(dict)
                            df = pd.concat([df, df1])

                        else:
                            print('Video numbers are not same in the QA data')
                            continue

        # append all the dataframes to a single dataframe

                #check if the video num is same in both the QA data
    # save the dataframe to a csv file
    dataframe_path = Path(f'{configs["train_dirpath1"]}/max_diff_frames.csv')
    df.to_csv(dataframe_path)

    return


def main():
    configs = {
        'train_dirpath1': Path('../../runs/training/train0014'),
        'train_dirpath2': Path('/media/kapilchoudhary/AILab_Server_harsha207/Harsha/21_DSSN/workspace/view_synthesis/research/011_SparseDepthPrior/runs/training/train0006'),
        'iter_num1': 30000,
        'iter_num2': 30000,
        'qa_metric_names': ['SSIM', 'PSNR', 'LPIPS_Alex']
    }
    max_QA_diff_frames(configs)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train-dirpath', nargs='+', type=str)
    #
    # args = parser.parse_args()
    # train_dirpaths = [Path(train_dirpath) for train_dirpath in args.train_dirpaths]
    # generate_videos(train_dirpaths)
    # return


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
