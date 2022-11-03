import cv2
import os
import argparse
import json


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--binary', action='store_true', help='Run binary testing on the videos')
    args = parser.parse_args()

    results = dict()

    for root, dirs, files in os.walk('./vids'):
        for file in files:
            path = os.path.join(root, file)
            scene_name = file

            results[file] = list()

            if args.binary:
                binary_check(path)

    if args.binary:
        for key in results.keys():
            res = results[key]
            results[key] = sum(res)/len(res)

        with open('results-binary.json', 'w') as f:
            json.dump(results, f, indent=4)
