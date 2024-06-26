# Shree KRISHNAya Namaha
# Moves data from train/test folders in runs folder to external HDD and creates a SymLink. git tracked files are
# retained in internal HDD. So, git need not follow symlinks.
# Extended from 19_SSLN/Py001/ExternalDataOrganizer01.py
# Author: Nagabhushan S N
# Last Modified: 27/12/2023

import os
import re
import shutil
import time
import datetime
import traceback
from typing import List

import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def move(src_path: Path, dest_path: Path, verbose: bool = True):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if verbose:
            print(f'moving {src_path.as_posix()} to {dest_path.as_posix()}')
        shutil.move(src_path, dest_path)
    except AttributeError as e1:
        print(src_path)
        print(dest_path)
        raise e1
    return


def move_dir(src_dirpath: Path, dest_dirpath: Path):
    if src_dirpath.exists():
        if src_dirpath.is_symlink():
            pass
        else:
            if dest_dirpath.exists():
                print(f'{src_dirpath} exists in both internal and external. Not moving.')
            else:
                move(src_dirpath, dest_dirpath)
    else:
        print(f'{src_dirpath} does not exist. Not moving.')
    return


def move_file(src_filepath: Path, dest_filepath: Path):
    if src_filepath.exists():
        if src_filepath.is_symlink():
            pass
        else:
            if dest_filepath.exists():
                print(f'{src_filepath} exists in both internal and external. Not moving.')
            else:
                move(src_filepath, dest_filepath)
    else:
        print(f'{src_filepath} does not exist. Not moving.')
    return


def create_symlink(src_path: Path, dest_path: Path, relative: bool = True, verbose: bool = True):
    if not relative:
        cmd = f'ln -s {src_path.absolute().as_posix()} {dest_path.parent.absolute().as_posix()}'
    else:
        cmd = f'ln -s {os.path.relpath(src_path, dest_path.parent)} {dest_path.parent.absolute().as_posix()}'
    if verbose:
        print(cmd)
    print(f'symlinking {src_path.as_posix()} to {dest_path.as_posix()}')
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    os.system(cmd)
    return


def symlink_dir(src_dirpath: Path, dest_dirpath: Path):
    if dest_dirpath.exists():
        if dest_dirpath.is_symlink():
            pass
        else:
            print(f'{src_dirpath} exists in both internal and external. Not symlinking this.')
    else:
        create_symlink(src_dirpath, dest_dirpath, relative=True, verbose=False)
    return


def symlink_file(src_filepath: Path, dest_filepath: Path):
    if dest_filepath.exists():
        if dest_filepath.is_symlink():
            pass
        else:
            print(f'{src_filepath} exists in both internal and external. Not symlinking this.')
    else:
        create_symlink(src_filepath, dest_filepath, relative=True, verbose=False)
    return


def exclusion_wrapper(fn, src_path: Path, dest_path: Path, excluded_paths: List[Path]):
    for excluded_path in excluded_paths:
        if src_path in excluded_path.parents:
            return
    fn(src_path, dest_path)
    return


def organize_workspace_data(external_dirpath: Path):
    int_dirpath = Path('../../../../workspace/')  # internal dirpath
    ext_dirpath = external_dirpath / 'workspace/'  # external dirpath
    move_workspace_data(int_dirpath, ext_dirpath)
    symlink_workspace_data(ext_dirpath, int_dirpath)
    return


def move_workspace_data(int_dirpath: Path, ext_dirpath: Path):
    for sl1_int_dirpath in sorted(int_dirpath.iterdir()):
        sl1_ext_dirpath = ext_dirpath / sl1_int_dirpath.stem
        for sl2_dirname in ['literature', 'research']:
            sl2_int_dirpath = sl1_int_dirpath / sl2_dirname
            sl2_ext_dirpath = sl1_ext_dirpath / sl2_dirname
            if sl2_int_dirpath.exists():
                for sl3_int_dirpath in sorted(sl2_int_dirpath.iterdir()):
                    if sl3_int_dirpath.stem == '000_Common':
                        continue

                    sl3_ext_dirpath = sl2_ext_dirpath / sl3_int_dirpath.stem

                    for sl4_dirname in ['data', 'pretrained_models']:
                        sl4_ext_dirpath = sl3_ext_dirpath / sl4_dirname
                        sl4_int_dirpath = sl3_int_dirpath / sl4_dirname
                        if sl4_int_dirpath.exists():
                            for sl5_int_dirpath in sorted(sl4_int_dirpath.iterdir()):
                                sl5_ext_dirpath = sl4_ext_dirpath / sl5_int_dirpath.stem
                                move_dir(sl5_int_dirpath, sl5_ext_dirpath)

                    for sl4_dirname in ['runs', 'runs/legacy']:
                        sl4_ext_dirpath = sl3_ext_dirpath / sl4_dirname
                        sl4_int_dirpath = sl3_int_dirpath / sl4_dirname
                        for sl5_dirname in ['training', 'testing']:
                            sl5_int_dirpath = sl4_int_dirpath / sl5_dirname
                            sl5_ext_dirpath = sl4_ext_dirpath / sl5_dirname
                            if sl5_int_dirpath.exists():
                                for sl6_int_dirpath in sorted(sl5_int_dirpath.iterdir()):  # train0001
                                    sl6_ext_dirpath = sl5_ext_dirpath / sl6_int_dirpath.stem

                                    skip_this = False
                                    for skip_dir_endings in [
                                        '.md'
                                    ]:
                                        if sl6_int_dirpath.as_posix().endswith(skip_dir_endings):
                                            skip_this = True
                                    if skip_this:
                                        print(f'Force Skipping: {sl6_int_dirpath.as_posix()}')
                                        continue

                                    for sl7_int_dirpath in sorted(sl6_int_dirpath.iterdir()):
                                        sl7_ext_dirpath = sl6_ext_dirpath / sl7_int_dirpath.name
                                        move_recursively(sl7_int_dirpath, sl7_ext_dirpath, ['quality_Scores', '__pycache__'], [], ['^events.out.tfevents.*$', '.*.mp4'])
    return


def move_recursively(sl7_int_dirpath: Path, sl7_ext_dirpath: Path, excluded_names: list, force_include_dirname_patterns: list, force_include_filename_patterns: list):
    if sl7_int_dirpath.is_dir() and (sl7_int_dirpath.name not in excluded_names):
        condition = (len(list(sl7_int_dirpath.rglob('**/*.json'))) == 0) and \
                    (len(list(sl7_int_dirpath.rglob('**/*.csv'))) == 0) and \
                    (len(list(sl7_int_dirpath.rglob('**/*.gin'))) == 0) and \
                    (len(list(sl7_int_dirpath.rglob('**/*.txt'))) == 0) and \
                    (len(list(sl7_int_dirpath.rglob('**/*.py'))) == 0)
        for pattern in force_include_dirname_patterns:
            condition = condition or (re.match(pattern, sl7_int_dirpath.name) is not None)
        if condition:
            move_dir(sl7_int_dirpath, sl7_ext_dirpath)
        else:
            for sl8_int_dirpath in sorted(sl7_int_dirpath.iterdir()):
                sl8_ext_dirpath = sl7_ext_dirpath / sl8_int_dirpath.name
                move_recursively(sl8_int_dirpath, sl8_ext_dirpath, excluded_names, force_include_dirname_patterns, force_include_filename_patterns)
    elif sl7_int_dirpath.is_file() and (sl7_int_dirpath.name not in excluded_names):
        condition = False
        for pattern in force_include_filename_patterns:
            condition = condition or (re.match(pattern, sl7_int_dirpath.name) is not None)
        if condition:
            move_file(sl7_int_dirpath, sl7_ext_dirpath)
    return 


def symlink_workspace_data(ext_dirpath: Path, int_dirpath: Path):
    for sl1_ext_dirpath in sorted(ext_dirpath.iterdir()):
        sl1_int_dirpath = int_dirpath / sl1_ext_dirpath.stem
        for sl2_dirname in ['literature', 'research']:
            sl2_ext_dirpath = sl1_ext_dirpath / sl2_dirname
            sl2_int_dirpath = sl1_int_dirpath / sl2_dirname
            if sl2_ext_dirpath.exists():
                for sl3_ext_dirpath in sorted(sl2_ext_dirpath.iterdir()):
                    sl3_int_dirpath = sl2_int_dirpath / sl3_ext_dirpath.stem

                    for sl4_dirname in ['data', 'pretrained_models']:
                        sl4_ext_dirpath = sl3_ext_dirpath / sl4_dirname
                        sl4_int_dirpath = sl3_int_dirpath / sl4_dirname
                        if sl4_ext_dirpath.exists():
                            for sl5_ext_dirpath in sorted(sl4_ext_dirpath.iterdir()):
                                sl5_int_dirpath = sl4_int_dirpath / sl5_ext_dirpath.stem
                                symlink_dir(sl5_ext_dirpath, sl5_int_dirpath)

                    for sl4_dirname in ['runs', 'runs/legacy']:
                        sl4_ext_dirpath = sl3_ext_dirpath / sl4_dirname
                        sl4_int_dirpath = sl3_int_dirpath / sl4_dirname
                        for sl5_dirname in ['training', 'testing']:
                            sl5_ext_dirpath = sl4_ext_dirpath / sl5_dirname
                            sl5_int_dirpath = sl4_int_dirpath / sl5_dirname
                            if sl5_ext_dirpath.exists():
                                for sl6_ext_dirpath in sorted(sl5_ext_dirpath.iterdir()):  # Train0001
                                    sl6_int_dirpath = sl5_int_dirpath / sl6_ext_dirpath.stem

                                    for sl7_ext_dirpath in sorted(sl6_ext_dirpath.iterdir()):
                                        sl7_int_dirpath = sl6_int_dirpath / sl7_ext_dirpath.name
                                        symlink_recursively(sl7_ext_dirpath, sl7_int_dirpath, ['quality_Scores', '__pycache__'], [], ['^events.out.tfevents.*$', '.*.mp4'])

    return


def symlink_recursively(sl7_ext_dirpath: Path, sl7_int_dirpath: Path, excluded_names: list, force_include_dirname_patterns: list, force_include_filename_patterns: list):
    if sl7_ext_dirpath.is_dir() and (sl7_ext_dirpath.name not in excluded_names) and (not sl7_int_dirpath.is_symlink()):
        condition = (len(list(sl7_int_dirpath.rglob('**/*.json'))) == 0) and \
                    (len(list(sl7_int_dirpath.rglob('**/*.csv'))) == 0) and \
                    (len(list(sl7_int_dirpath.rglob('**/*.gin'))) == 0) and \
                    (len(list(sl7_int_dirpath.rglob('**/*.txt'))) == 0) and \
                    (len(list(sl7_int_dirpath.rglob('**/*.py'))) == 0)
        for pattern in force_include_dirname_patterns:
            condition = condition or (re.match(pattern, sl7_ext_dirpath.name) is not None)
        if condition:
            symlink_dir(sl7_ext_dirpath, sl7_int_dirpath)
        else:
            for sl8_ext_dirpath in sorted(sl7_ext_dirpath.iterdir()):
                sl8_int_dirpath = sl7_int_dirpath / sl8_ext_dirpath.name
                symlink_recursively(sl8_ext_dirpath, sl8_int_dirpath, excluded_names, force_include_dirname_patterns, force_include_filename_patterns)
    elif sl7_ext_dirpath.is_file() and (sl7_ext_dirpath.name not in excluded_names) and (not sl7_int_dirpath.is_symlink()):
        condition = False
        for pattern in force_include_filename_patterns:
            condition = condition or (re.match(pattern, sl7_ext_dirpath.name) is not None)
        if condition:
            symlink_file(sl7_ext_dirpath, sl7_int_dirpath)
    return 


def gitignore_symlinks():
    project_dirpath = Path('../../../../').absolute()
    gitignore_path = project_dirpath / '.gitignore'
    with open(gitignore_path.as_posix(), 'r') as gitignore_file:
        gitignore_contents = gitignore_file.readlines()
    empty_line_index = gitignore_contents.index('\n')
    gitignore_contents = gitignore_contents[:empty_line_index+1]
    with open(gitignore_path.as_posix(), 'w') as gitignore_file:
        gitignore_file.writelines(gitignore_contents)

    os.chdir(project_dirpath)
    cmd = 'find workspace/* -type l >> .gitignore'
    os.system(cmd)

    # Sort the lines
    with open(gitignore_path.as_posix(), 'r') as gitignore_file:
        gitignore_contents = gitignore_file.readlines()
    empty_line_index = gitignore_contents.index('\n')
    gitignore_contents1 = gitignore_contents[:empty_line_index+1]
    gitignore_contents2 = gitignore_contents[empty_line_index+1:]
    with open(gitignore_path.as_posix(), 'w') as gitignore_file:
        gitignore_file.writelines(gitignore_contents1)
        gitignore_file.writelines(sorted(gitignore_contents2))
    return


def demo1():
    # external_dirpath = Path('../../../../../../../../../VSTURW01/SNB/21_DSSN')
    external_dirpath = Path(''.join(len(Path('.').absolute().relative_to(Path(f"/media/{os.environ['USER']}")).parents) * ['../'])) / 'VSTURW01/SNB/21_DSSN'
    # external_dirpath = Path(''.join((len(Path('.').absolute().parents) - 2) * ['../'])) / 'VSTURW01/SNB/21_DSSN'
    organize_workspace_data(external_dirpath)
    gitignore_symlinks()
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

    from snb_utils import Telegrammer

    time.sleep(5)
    message_content = f'R21/Py001/{this_filename} has finished.\n' + run_result
    Telegrammer.send_message(message_content, chat_names=['Nagabhushan'])
