# Shree KRISHNAya Namaha
# Creates symlinks for the paper folders
# Author: Nagabhushan S N
# Last Modified: 21/12/2022

import os.path
import shutil
from pathlib import Path


class SymlinkCreator:
    def __init__(self, src_literature_path: Path, tgt_literature_path: Path, papers_list_path: Path):
        self.src_literature_path = src_literature_path
        self.tgt_literature_path = tgt_literature_path
        self.papers_list_path = papers_list_path
        self.papers_list = []
        return

    def read_papers_list(self):
        with open(self.papers_list_path.as_posix(), 'r') as papers_list_file:
            papers_list = [line.strip() for line in papers_list_file.readlines()]
        self.papers_list = sorted(papers_list)
        with open(self.papers_list_path.as_posix(), 'w') as papers_list_file:
            papers_list_file.writelines([line + '\n' for line in self.papers_list])
        return

    def create_symlinks(self):
        for paper_name in self.papers_list:
            tgt_dirname = self.tgt_literature_path / paper_name
            src_dirname = self.src_literature_path / paper_name

            if tgt_dirname.is_symlink():
                if tgt_dirname.exists():
                    # Link already exists
                    continue
                else:
                    # Broken link
                    tgt_dirname.unlink()
            cmd = f'ln -s "{os.path.relpath(src_dirname, tgt_dirname.parent)}" "{tgt_dirname.parent.as_posix()}"'
            os.system(cmd)
        return


def demo1():
    src_literature_path = Path('../../../Resources/Literature').absolute()
    tgt_literature_path = Path('../data').absolute()
    papers_list_path = Path('../res/Papers.txt')

    tgt_literature_path.mkdir(parents=True, exist_ok=True)
    symlink_creator = SymlinkCreator(src_literature_path, tgt_literature_path, papers_list_path)
    symlink_creator.read_papers_list()
    symlink_creator.create_symlinks()
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    main()