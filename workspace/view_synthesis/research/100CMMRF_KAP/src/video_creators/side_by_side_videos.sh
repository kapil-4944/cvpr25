#!/bin/bash
# Shree KRISHNAya Namaha

scene_names=(
        "coffee_martini"
        "cook_spinach"
        "cut_roasted_beef"
        "flame_salmon_1"
        "flame_steak"
        "sear_steak"
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
	videos1_dirpath="/mnt/nagasn51/Harsha/21_DSSN/workspace/view_synthesis/research/100CMMRF_KAP/runs/training/train0016/${scene_name}/predicted_videos_iter030000/rgb"
	videos2_dirpath="/mnt/nagasn51/Harsha/21_DSSN/workspace/view_synthesis/research/012_DifferentCameraIntrinsics/runs/training/train0052/${scene_name}/predicted_videos_iter030000/rgb"

	output_dirpath="/home/ece/Desktop/Kapil/Set03"
	mkdir -p ${output_dirpath}

#	video1_path="${videos1_dirpath}/0000/0000.mp4"
#	video2_path="${videos2_dirpath}/0000.mp4"
#	output_path="${output_dirpath}/${scene_name}.mp4"
#	ffmpeg -i ${video1_path} -i ${video2_path} -filter_complex hstack ${output_path}

	video1_path="${videos1_dirpath}/0000_spiral01.mp4"
	video2_path="${videos2_dirpath}/0000_spiral01.mp4"
	output_path="${output_dirpath}/${scene_name}_spiral01.mp4"
	ffmpeg -i ${video1_path} -i ${video2_path} -filter_complex hstack ${output_path}
done