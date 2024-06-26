# Shree KRISHNAya Namaha
# Merges all Notes in Data
# Author: Nagabhushan S N
# Last Modified: 10/02/2020

import glob
import os
import re
from pathlib import Path
from typing import List


def create_all_notes(literature_dirpath: Path, skeleton_filepath: Path, excluded_dirnames: List[str], output_filepath: Path):
    paper_names = [paper_dirpath.as_posix() for paper_dirpath in literature_dirpath.iterdir()
                   if (paper_dirpath.is_dir() and (paper_dirpath.as_posix() not in excluded_dirnames))]
    create_notes(literature_dirpath, paper_names, skeleton_filepath, output_filepath)
    return


def create_notes(literature_dirpath: Path, papers_list: list, skeleton_filepath, output_filepath: Path):
    tex_filepaths = []
    [tex_filepaths.extend(literature_dirpath.rglob(f'{glob.escape(paper)}/Notes/*.tex')) for paper in papers_list]

    content_dict = {}
    for tex_filepath in tex_filepaths:
        # print(f'Reading paper: {tex_filepath.parent.parent.name}')
        with open(tex_filepath.as_posix(), 'r') as tex_file:
            content = tex_file.read()
            pattern = r'\\title{(.+) - Notes}[\s\S]+\\begin{document}\n([\s\S]+)\\end{document}'
            match = re.search(pattern, content)
            if match:
                title = match.group(1)
                label = get_paper_label(title)
                body = match.group(2)
                body = body.replace('section', 'subsection')
                body = body.replace(r'\label{subsec:', r'\label{subsubsec:')
                body = body.replace(r'\label{sec:', r'\label{subsec:')
                body = body.replace('paragraph', 'subparagraph')
                body = body.replace('subsubsubsection', 'paragraph')
                body = body.replace('\\maketitle\n', '')
                body = body.replace('\\pdfbookmark[1]{\\contentsname}{toc}\n', '')
                body = body.replace('\\tableofcontents\n\\newpage\n\n', '')
                body = body.replace('\\tableofcontents\n\\newpage\n', '')
                body = re.sub(r'\s*\\tableofcontents\n\s*\\newpage\n\n', '', body)
                body = body.replace('\\pagenumbering{gobble}\n', '')
                body = body.replace('\\pagenumbering{arabic}\n\n', '')
                body = body.replace('\\pagenumbering{arabic}\n', '')
                body = body.replace('\\pdfbookmark[1]{Abstract}{abstract}\n', '')
                body = re.sub(r'(label{.+?:)', fr'\1{label}:', body)
            else:
                raise Exception('No match found.')
            content_dict[title] = body

    merged_content = ''
    for title, body in content_dict.items():
        label = get_paper_label(title)
        merged_content += f'\\section{{{title}}}\\label{{sec:{label}}}\n{body}\\newpage\n\n'

    with open(skeleton_filepath, 'r') as skeleton_file:
        skeleton = skeleton_file.read()
    final_content = skeleton.replace('<NotesMergerContent>', merged_content)

    # output_filepath = output_filepath / notes_name / f'Literature - {notes_name}.tex'
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath.as_posix(), 'w+') as notes_file:
        notes_file.write(final_content)
    run_latex(output_filepath)
    return


def get_paper_label(title: str):
    label = title.replace(' - ', '_').replace(' -', '_').replace('-', '_')
    label = label.replace(' : ', '_').replace(': ', '_').replace(' ', '_')
    return label


def run_latex(tex_filepath: Path):
    cmd = f'pdflatex -output-directory "{tex_filepath.parent.as_posix()}" "{tex_filepath.as_posix()}"'
    execute_cmd(cmd)
    execute_cmd(cmd)
    return


def execute_cmd(cmd):
    print(cmd)
    os.system(cmd)


def main():
    literature_dirpath = Path('../data')
    skeleton_filepath = Path('../res/NotesSkeleton.txt')
    excluded_dirnames = []
    output_filepath = Path('../combined_notes/MyNotes.tex')
    create_all_notes(literature_dirpath, skeleton_filepath, excluded_dirnames, output_filepath)
    return


if __name__ == '__main__':
    main()
