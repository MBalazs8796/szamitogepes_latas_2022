import cv2
import os
import argparse
import json

import numpy.linalg
import numpy as np

import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import match_mover


def binary_check(video):
    global scene_name, results

    cv2.namedWindow(scene_name)
    capture = cv2.VideoCapture(video)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            cv2.destroyWindow(scene_name)
            break

        cv2.imshow(scene_name, frame)
        key = cv2.waitKey(0)
        if key == ord('y'):
            results[scene_name].append(1)
        elif key == ord('x'):
            results[scene_name].append(0)


def get_camera_matrix(config):
    elements = dict()
    with open(f'../orbslam_driver/{config}.yaml', 'r') as f:
        for line in f.readlines():
            if line.startswith('Camera.fx:'):
                elements['fx'] = float(line.strip().split(' ')[-1])
            elif line.startswith('Camera.fy:'):
                elements['fy'] = float(line.strip().split(' ')[-1])
            elif line.startswith('Camera.cx:'):
                elements['cx'] = float(line.strip().split(' ')[-1])
            elif line.startswith('Camera.cy:'):
                elements['cy'] = float(line.strip().split(' ')[-1])

    K = np.array([[elements['fx'], 0, elements['cx']],
                  [0, elements['fy'], elements['cy']],
                  [0, 0, 1]])
    K = np.hstack([K, np.zeros([3, 1])])
    K = np.vstack([K, [0, 0, 0, 1]])

    return K


def get_results(video_name):
    with open(f'../orbslam_driver/extracted/{video_name}/result.json', 'r') as f:
        return json.load(f)


def placement_check(video_name, config):
    poses, has_pose, names = match_mover.read_kerframe_trajectory(f'../orbslam_driver/extracted/{video_name}/KeyFrameTrajectory.txt', f'../orbslam_driver/extracted/{video_name}/rgb.txt')

    n_frames = len(poses)
    K = get_camera_matrix(config)
    sm = get_results(video_name)

    norms = list()

    for i in range(n_frames):
        t_model = [[sm[0]['h'] / -5], [sm[0]['w'] / 10], [-500]]
        R_model = match_mover.degree2R(sm[0]['roll'], sm[0]['pitch'] / 2, sm[0]['yaw'] / 2)
        Rt_model = np.hstack([R_model, t_model])
        Rt_model = np.vstack([Rt_model, [0, 0, 0, 1]])

        Rt_cam = poses[i]

        Rt = Rt_cam @ Rt_model

        Rt = match_mover.approx_rotation(Rt[:-1, :])

        P = K @ Rt

        try:
            R = match_mover.degree2R(sm[i + 1]['roll'], sm[i + 1]['pitch'], sm[i + 1]['yaw'])
            R = np.hstack([R, [[sm[i + 1]['h']], [sm[i + 1]['w']], [sm[i + 1]['scale']]]])
            R = np.vstack([R, [0, 0, 0, 1]])
        except IndexError:
            continue

        norms.append(numpy.linalg.norm(P - R, 'fro'))

    #mse = np.mean([norm ** 2 for norm in norms])

    result = dict()
    result['norms'] = norms
    #result['mse'] = mse

    with open(f'results_{video_name}.json', 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-b', '--binary', action='store_true', help='Run binary testing on the videos')
    group.add_argument('-v', '--video', metavar=None, help='Run a placement test on a specific video')

    parser.add_argument('-c', '--camera', metavar=None, help='The camera config name with the appropriate camera parameters')
    args = parser.parse_args()

    results = dict()

    if args.binary:
        for root, dirs, files in os.walk('./vids'):
            for file in files:
                path = os.path.join(root, file)
                scene_name = file

                results[file] = list()
                binary_check(path)

        for key in results.keys():
            res = results[key]
            results[key] = np.mean(res)

        with open('results-binary.json', 'w') as f:
            json.dump(results, f, indent=4)

    if args.video:
        placement_check(args.video, args.camera)
