#!/bin/bash
# Shree KRISHNAya Namaha

scene_names=(
        "coffee_martini"
#        "cook_spinach"
#        "cut_roasted_beef"
#        "flame_salmon_1"
#        "flame_steak"
#        "sear_steak"
        )
#scene_name=(
#                "Birthday"
#                "Theartre"
#                "Train"
#                "Painter"
#            )

# Loop over every scene and run training and rendering
for scene_name in ${scene_names[@]};
do
	videos1_dirpath="/mnt/nagasn51/Harsha/21_DSSN/workspace/view_synthesis/research/100CMMRF_KAP/runs/training/train0014/${scene_name}/predicted_videos_iter030000/rgb"
#	videos2_dirpath="/media/kapilchoudhary/harsha_workstation/Harsha/21_DSSN/workspace/view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train1025/${scene_name}/predicted_videos_iter030000/rgb"

#	output_dirpath=videos1_dirpath
#	mkdir -p ${output_dirpath}

	video1_path="${videos1_dirpath}/0000"
	ffmpeg_arg = "${video1_path}/%04d.png"
#	video2_path="${videos2_dirpath}/0000"
	output_path="${videos1_dirpath}/0000.mp4"
#	ffmpeg -framerate 24 -i Project%03d.png Project.mp4
	ffmpeg -framerate 30 -i ${ffmpeg_arg}  ${output_path}

done