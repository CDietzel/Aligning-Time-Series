import pickle
from itertools import chain, repeat, tee
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
from tslearn.metrics import ctw

from Code.GDTW import gromov_dtw


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as file:
        return pickle.load(file)


def extract_human_keypoints(poses):
    keypoints_list = []
    for pose in poses:
        if pose is not None:
            keypoints_list.append(pose.landmarks_world)  # Only use this,
            # keypoints_list.append(pose.norm_landmarks)  # Not these.
            # keypoints_list.append(pose.landmarks)  # They are image-relative
    keypoints = np.array(keypoints_list)[:, 0:33, :]
    return keypoints


pre_swap = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]
post_swap = [
    0,
    4,
    5,
    6,
    1,
    2,
    3,
    8,
    7,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
    18,
    17,
    20,
    19,
    22,
    21,
    24,
    23,
    26,
    25,
    28,
    27,
    30,
    29,
    32,
    31,
]


# UPDATE THESE PATHS
model_output_folder = Path("/home/locobot/Documents/Repos/ibc/ibc/output")
robot_test_motion_folder = Path("/home/locobot/Documents/Repos/depthai_blazepose/5DoF")
human_motion_folder = Path("/home/locobot/Documents/Repos/depthai_blazepose/outputs/")
loss_output = Path("/home/locobot/Documents/Repos/Aligning-Time-Series/output")

robot_motion_prefix = "oracle_interbotix_"
robot_motion_postfix = "_n"
robot_motion_extension = ".modulated"
robot_test_motion_extension = ".recording"
human_motion_extension = ".pickle"
motion_name_list = ["test1", "test2", "test3", "test4", "test5"]

tag_list = ["ibc_langevin_test", "mse_test"]
num_seeds = 3
num_hyperparameters = [18, 18]

sum_hyperparameters = sum(num_hyperparameters)
total_hyperparameters = num_seeds * sum_hyperparameters

Gdtw = gromov_dtw(max_iter=10, gamma=0.1, loss_only=1, dtw_approach="GDTW", verbose=0)
Soft_Gdtw = gromov_dtw(
    max_iter=10, gamma=0.1, loss_only=1, dtw_approach="soft_GDTW", verbose=0
)

full_ctw_scores = []
full_soft_gdtw_scores = []
full_gdtw_scores = []
full_nums = []
full_tags = []
full_names = []

# avg_ctw_scores = []
# avg_soft_gdtw_scores = []
# avg_gdtw_scores = []
# avg_nums = []
# avg_tags = []
# avg_names = []

for name in motion_name_list:
    i = -1
    tag = "human"

    robot_motion_path = robot_test_motion_folder / (name + robot_test_motion_extension)
    human_motion_path = human_motion_folder / (name + human_motion_extension)

    human_poses = load_pickle(human_motion_path)
    robot_data = load_pickle(robot_motion_path)[0][::5]
    human_data = extract_human_keypoints(human_poses)

    # Only include arm keypoints (11-22)
    human_data = human_data[:, 11:23, :]

    # Reshape to flatten xyz coordinates for all keypoints for each frame
    human_data = human_data.reshape(-1, 36)

    # Ignore first few samples (noisy data)
    human_data = human_data[3:]

    ctw_loss = ctw(robot_data, human_data)

    robot_data = torch.tensor(robot_data)
    human_data = torch.tensor(human_data)

    gdtw_loss = Gdtw(robot_data, human_data).item()
    soft_gdtw_loss = Soft_Gdtw(robot_data, human_data).item()

    #
    #
    #
    #
    #
    #

    full_ctw_scores.append(ctw_loss)
    full_gdtw_scores.append(gdtw_loss)
    full_soft_gdtw_scores.append(soft_gdtw_loss)
    full_nums.append(i)
    full_tags.append(tag)
    full_names.append(name)

    print("score for num " + str(i) + "/" + name + " is: " + str(gdtw_loss))


for i, tag in enumerate(
    chain.from_iterable(
        tee(
            chain.from_iterable(
                repeat(x[1], x[0]) for x in zip(num_hyperparameters, tag_list)
            ),
            num_seeds,
        )
    )
):
    # temp_ctw_scores = []
    # temp_soft_gdtw_scores = []
    # temp_gdtw_scores = []
    robot_motion_folder = model_output_folder / str(tag) / str(i)
    for name in motion_name_list:
        robot_motion_path = robot_motion_folder / (
            robot_motion_prefix + name + robot_motion_postfix + robot_motion_extension
        )
        human_motion_path = human_motion_folder / (name + human_motion_extension)

        human_poses = load_pickle(human_motion_path)
        robot_data = load_pickle(robot_motion_path)
        human_data = extract_human_keypoints(human_poses)

        # Only include arm keypoints (11-22)
        human_data = human_data[:, 11:23, :]

        # Reshape to flatten xyz coordinates for all keypoints for each frame
        human_data = human_data.reshape(-1, 36)

        # Ignore first few samples (noisy data)
        human_data = human_data[3:]

        ctw_loss = ctw(robot_data, human_data)

        robot_data = torch.tensor(robot_data)
        human_data = torch.tensor(human_data)

        gdtw_loss = Gdtw(robot_data, human_data).item()
        soft_gdtw_loss = Soft_Gdtw(robot_data, human_data).item()

        #
        #
        #
        #
        #
        #

        full_ctw_scores.append(ctw_loss)
        full_gdtw_scores.append(gdtw_loss)
        full_soft_gdtw_scores.append(soft_gdtw_loss)
        full_nums.append(i)
        full_tags.append(tag)
        full_names.append(name)

        print("score for num " + str(i) + "/" + name + " is: " + str(gdtw_loss))

        # temp_ctw_scores.append(ctw_loss)
        # temp_gdtw_scores.append(gdtw_loss)
        # temp_soft_gdtw_scores.append(soft_gdtw_loss)

    # avg_ctw_scores.append(mean(temp_ctw_scores))
    # avg_gdtw_scores.append(mean(temp_gdtw_scores))
    # avg_soft_gdtw_scores.append(mean(temp_soft_gdtw_scores))
    # avg_nums.append(i)
    # avg_tags.append(tag)
    # print("score for num " + str(i) + " is: " + str(mean(temp_gdtw_scores)))

full_results = pd.DataFrame(
    {
        "Model ID": full_nums,
        "Model Type": full_tags,
        "Test File": full_names,
        "CTW Scores": full_ctw_scores,
        "GDTW Scores": full_gdtw_scores,
        "Soft GDTW Scores": full_soft_gdtw_scores,
    }
)
# avg_results = pd.DataFrame(
#     {
#         "Model ID": avg_nums,
#         "Model Type": avg_tags,
#         "CTW Scores": avg_ctw_scores,
#         "GDTW Scores": avg_gdtw_scores,
#         "Soft GDTW Scores": avg_soft_gdtw_scores,
#     }
# )

loss_output.mkdir(parents=True, exist_ok=True)

full_results.to_csv(loss_output / "gdtw_full_results.csv", index=False)
# avg_results.to_csv(loss_output / "avg_results.csv", index=False)

# avg_results[["CTW Scores", "GDTW Scores", "Soft GDTW Scores"]] = avg_results.groupby(
#     avg_results.index % sum_hyperparameters
# )[["CTW Scores", "GDTW Scores", "Soft GDTW Scores"]].sum(numeric_only=True)

# avg_results.drop(
#     avg_results.tail(sum_hyperparameters * (num_seeds - 1)).index, inplace=True
# )

# avg_results[["CTW Scores", "GDTW Scores", "Soft GDTW Scores"]] /= 3

# avg_results.to_csv(loss_output / "avg_avg_results.csv", index=False)

# for tag_num, tag in enumerate(tag_list):
#     filtered_avg = avg_results.loc[avg_results["Model Type"] == tag]

#     best_ctw = filtered_avg[
#         filtered_avg["CTW Scores"] == filtered_avg["CTW Scores"].min()
#     ]
#     best_gdtw = filtered_avg[
#         filtered_avg["GDTW Scores"] == filtered_avg["GDTW Scores"].min()
#     ]
#     best_soft_gdtw = filtered_avg[
#         filtered_avg["Soft GDTW Scores"] == filtered_avg["Soft GDTW Scores"].min()
#     ]
#     best_ctw.to_csv(loss_output / ("best_ctw_" + str(tag) + ".csv"), index=False)
#     best_gdtw.to_csv(loss_output / ("best_gdtw_" + str(tag) + ".csv"), index=False)
#     best_soft_gdtw.to_csv(
#         loss_output / ("best_soft_gdtw_" + str(tag) + ".csv"), index=False
#     )
