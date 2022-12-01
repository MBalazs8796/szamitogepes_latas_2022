import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tqdm


@dataclass(frozen=True, order=True)
class KeyFrameTrajectoryFileInfo:
    """frame times are stored as negative values to achieve simple larger is better comparison"""
    first_frame_time: float = field(init=False)
    frame_count: int = field(init=False)
    frame_times: list[int] = field(hash=False)
    sha256_hash: str = field(init=False)

    def __post_init__(self):
        frame_count = len(self.frame_times)
        object.__setattr__(self, 'frame_count', frame_count)
        first_frame_time = self.frame_times[0] if self.frame_count > 0 else -(sys.maxsize - 1)
        object.__setattr__(self, 'first_frame_time', first_frame_time)
        sha256_hash = hashlib.sha256(json.dumps(self.frame_times).encode()).hexdigest()
        object.__setattr__(self, 'sha256_hash', sha256_hash)

    def get_keyframe_filename(self):
        return f'KeyFrameTrajectory.hash_v1.{self.sha256_hash}.txt'

    def get_trajectory_filename(self):
        return f'Trajectory.hash_v1.{self.sha256_hash}.txt'

    @staticmethod
    def from_file(filename: str):
        frame_times: list[int] = []
        with open(filename,"r") as f:
            for raw_line in f.readlines():
                line_parts = raw_line.split(' ')
                if len(line_parts) > 0:
                    # round numbers to limit number of results that would count as unique
                    # store frame time as negative value
                    frame_time = -int(np.around(float(line_parts[0])))
                    frame_times.append(frame_time)

        return KeyFrameTrajectoryFileInfo(frame_times=frame_times)


class OrbSlamRunner:
    ORB_SLAM_HOME = '/home/ORB_SLAM2'
    TUM_DRIVER = 'Examples/Monocular/mono_tum'
    TUM_DRIVER_LOCALIZATION = 'Examples/Monocular/mono_tum_local'
    VOCABULARY = 'Vocabulary/ORBvoc.txt'

    duplicate_results: int
    """number of times an already existing result was provided"""

    def __init__(self, video_name: str, config: str, localization: bool, repeat: int):
        self.video_name = video_name
        self.video_folder = _get_video_folder(video_name)
        self.config = config
        self.repeat = repeat
        self.tum_driver = OrbSlamRunner.TUM_DRIVER_LOCALIZATION if localization else OrbSlamRunner.TUM_DRIVER
        self.keyframe_trajectories_dir = f'{self.video_folder}/keyframe_trajectories'
        self.duplicate_results = 0

        Path(self.keyframe_trajectories_dir).mkdir(parents=True, exist_ok=True)

    def _orb_slam_command_args(self):
        return [
          f'{OrbSlamRunner.ORB_SLAM_HOME}/{self.tum_driver}',
          f'{OrbSlamRunner.ORB_SLAM_HOME}/{OrbSlamRunner.VOCABULARY}',
          self.config,
          f'{self.video_folder}/'
        ]

    def _run_orb_slam(self):
        subprocess.run(self._orb_slam_command_args())
        original_keyframe_filename = './KeyFrameTrajectory.txt'
        file_info = KeyFrameTrajectoryFileInfo.from_file(original_keyframe_filename)
        shutil.move(original_keyframe_filename, f'{self.keyframe_trajectories_dir}/{file_info.get_keyframe_filename()}')
        if os.access('./Trajectory.txt', os.W_OK):
            shutil.move('./Trajectory.txt', f'{self.keyframe_trajectories_dir}/{file_info.get_trajectory_filename()}')
        else:
            print('WARN: No Trajectory.txt file found')
        return file_info

    def run(self):
        """
        Runs ORBSLAM multiple times until we get a specified amount of duplicate results
        """
        keyframe_trajectory_files: set[KeyFrameTrajectoryFileInfo] = set()

        for root, _, files in os.walk(self.keyframe_trajectories_dir):
            print('Load existing (KeyFrame)Trajectory files')
            for file in tqdm.tqdm(files):
                if not file.startswith('KeyFrameTrajectory'):
                    continue
                filename = os.path.join(root, file)
                file_info = KeyFrameTrajectoryFileInfo.from_file(filename)
                keyframe_trajectory_files.add(file_info)

        counter = 0
        while counter < self.repeat:
            current_file_info = self._run_orb_slam()
            if current_file_info in keyframe_trajectory_files:
                self.duplicate_results += 1
            keyframe_trajectory_files.add(current_file_info)
            counter += 1
            print(f'Status report - {self.video_name}')
            print(current_file_info)
            print(f'Duplicate results: {self.duplicate_results}')
            print(f'Run count: {counter}/{self.repeat}')

        best_result_file_info = max(keyframe_trajectory_files)
        shutil.copy(f'{self.keyframe_trajectories_dir}/{best_result_file_info.get_keyframe_filename()}', f'{self.video_folder}/KeyFrameTrajectory.txt')
        shutil.copy(f'{self.keyframe_trajectories_dir}/{file_info.get_trajectory_filename()}', f'{self.video_folder}/Trajectory.txt')


def _get_video_folder(video_name):
    video_folder = f'./extracted/{video_name}'
    Path(video_folder).mkdir(parents=True, exist_ok=True)
    return video_folder


def extract_frames(video):
    print(f'Extracting frames from {video}')
    extracted = list()

    capture = cv2.VideoCapture(video)

    i = 1.0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        extracted.append((frame, round(i, 2)))
        i += 0.05

    return extracted


def save_data(video, video_name):
    print(f'Creating folder for {video}')

    video_folder = _get_video_folder(video_name)
    Path(f'{video_folder}/rgb').mkdir(parents=True, exist_ok=True)

    with open(f'{video_folder}/rgb.txt', 'w') as f:
        f.write('# whatever 1\n')
        f.write('# whatever 2\n')
        f.write('# whatever 3\n')

        for data in tqdm.tqdm(extract_frames(video)):
            f.write(f'{data[1]} rgb/{data[1]}.png\n')
            cv2.imwrite(f'{video_folder}/rgb/{data[1]}.png', data[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', metavar=None, help='The config file to use')
    parser.add_argument('-l', '--localization', action='store_true', help='Run ORB_SLAM in localization mode')
    parser.add_argument('-r', '--repeat', metavar=None, default=20, type=int, help='Execute analysis this many times before returning final result')
    args = parser.parse_args()

    for root, dirs, files in os.walk('./vids'):
        for file in files:
            video = os.path.join(root, file)
            video_name = file.split('.')[0]
            # save_data(video, video_name)
            if args.config:
                runner = OrbSlamRunner(
                    video_name,
                    args.config,
                    args.localization,
                    args.repeat
                )
                runner.run()
