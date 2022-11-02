import cv2
import os
import subprocess
import shutil
import argparse
import tqdm
import time


ORB_SLAM_HOME = '/home/ORB_SLAM2'
TUM_DRIVER = 'Examples/Monocular/mono_tum'
VOCABULARY = 'Vocabulary/ORBvoc.txt'


def extract_frames(video):
    print(f'Extracting frames from {video}')
    extracted = list()

    capture = cv2.VideoCapture(video)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        extracted.append((frame, time.time()))

    return extracted


def save_data(video, video_name):
    print(f'Creating folder for {video}')

    if not os.path.isdir('./extracted'):
        os.mkdir('./extracted')

    video_folder = f'./extracted/{video_name}'
    if not os.path.isdir(video_folder):
        os.mkdir(video_folder)
        os.mkdir(f'{video_folder}/rgb')

    with open(f'{video_folder}/rgb.txt', 'w') as f:
        f.write('# whatever 1\n')
        f.write('# whatever 2\n')
        f.write('# whatever 3\n')

        for data in tqdm.tqdm(extract_frames(video)):
            f.write(f'{data[1]} rgb/{data[1]}.png\n')
            cv2.imwrite(f'{video_folder}/rgb/{data[1]}.png', data[0])


def run_orb_slam(video_name, config):
    print('Running ORB_SLAM')
    video_folder = f'./extracted/{video_name}'
    print(f'{ORB_SLAM_HOME}/{TUM_DRIVER} {ORB_SLAM_HOME}/{VOCABULARY} {config} {video_folder}/')
    subprocess.run(f'{ORB_SLAM_HOME}/{TUM_DRIVER} {ORB_SLAM_HOME}/{VOCABULARY} {config} {video_folder}/', shell=True)
    shutil.move('./KeyFrameTrajectory.txt', f'{video_folder}/KeyFrameTrajectory.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', metavar=None, help='The config file to use')
    args = parser.parse_args()

    for root, dirs, files in os.walk('./vids'):
        for file in files:
            video = os.path.join(root, file)
            video_name = file.split('.')[0]
            save_data(video, video_name)
            run_orb_slam(video_name, args.config)
