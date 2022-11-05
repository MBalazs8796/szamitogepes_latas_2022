import numpy as np
import math
import cv2
import json
import argparse
from copy import deepcopy
from object_loader import OBJ


def _get_result_file():
    global scene_name
    return f'../orbslam_driver/extracted/{scene_name}/result.json'


def degree2R(roll, pitch, yaw):
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    yawMatrix = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix

    return R


def render(img, obj, projection, h, w, color=False, scale=1):
    """
    Render a loaded obj model into the current video frame
    Input:
      img: bg image
      obj: loaded 3d model
      projection: 4x4 transformation matrix
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3, dtype='int32') * scale
    # h, w = model.shape

    img = deepcopy(img)

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])

        points = np.dot(points, scale_matrix)

        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)

        imgpts = np.int32(dst)
        r = list()
        for f in imgpts:
            r.append(list())
            for s in f:
                r[-1].append(s[:-1])
        imgpts = np.array(r)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = cv2.hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)
    #print('Render Success')

    return img


def _use_placement_from_file():
    global index, h, w, scale, roll, pitch, yaw, placements_from_file

    if len(placements_from_file) == 0:
        return
    previous_psotion = placements_from_file[index]

    # this causes multiple image updates
    cv2.setTrackbarPos('h', scene_name, previous_psotion['h'])
    cv2.setTrackbarPos('w', scene_name, previous_psotion['w'])
    cv2.setTrackbarPos('scale', scene_name, previous_psotion['scale'])
    cv2.setTrackbarPos('roll', scene_name, previous_psotion['roll'])
    cv2.setTrackbarPos('pitch', scene_name, previous_psotion['pitch'])
    cv2.setTrackbarPos('yaw', scene_name, previous_psotion['yaw'])


def showimg():
    global h, w, scale, roll, pitch, yaw,scene_name, og_img

    R = degree2R(roll, pitch, yaw)
    R = np.hstack([R, [[h],[w],[scale]]])
    R = np.vstack([R, [0,0,0,1]])
    img = render(og_img, obj, R, h=1, w=1, color=False, scale = scale)
    cv2.imshow(scene_name, img)

def roll_rot_track(r):
    global roll
    
    roll = r

    showimg()

def pitch_rot_track(p):
    global pitch
    
    pitch = p

    showimg()
    
def yaw_rot_track(y):
    global yaw
    
    yaw = y

    showimg()

def vertical_track(s_poz):
    global h

    h = s_poz
    showimg()

def horizontal_track(s_poz):
    global w

    w = s_poz
    showimg()

def scale_track(sc):
    global scale

    scale = sc
    showimg()

def getFirstFrame(videofile: str):
    return cv2.imread(f'../orbslam_driver/extracted/{videofile}/rgb/1.0.png')

def save(*args):
    global h, w, scale, roll, pitch, yaw, results, scene_name
    results.append({
        'h' : h,
        'w' : w,
        'scale' : scale,
        'roll' : roll,
        'pitch' : pitch,
        'yaw' : yaw
    })

    if not next_image():
        with open(_get_result_file(), 'w') as f:
            json.dump(results, f, indent=4)
        cv2.destroyAllWindows()
        print('Placement complete')
        exit(0)


def get_keyframes(video_name):
    keyframes = list()
    with open(f'../orbslam_driver/extracted/{video_name}/KeyFrameTrajectory.txt', 'r') as f:
        for line in f.readlines():
            timestamp = float(line.split(' ')[0])
            timestamp = round(timestamp, 2)
            keyframes.append(cv2.imread(f'../orbslam_driver/extracted/{video_name}/rgb/{timestamp}.png'))

    return keyframes


def next_image():
    global index, og_img, keyframes

    if index == len(keyframes):
        return False

    og_img = keyframes[index]
    index += 1
    _use_placement_from_file()
    showimg()

    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', metavar=None, help='The name of the video')
    parser.add_argument('-p', '--previous', action='store_true', help='Use previous object placement file')
    args = parser.parse_args()

    scene_name = args.video

    index = 0
    results = list()
    keyframes = get_keyframes(args.video)

    obj = OBJ('lego.obj', swapyz=True)
    og_img = getFirstFrame(args.video)
    
    og_img_height, og_img_width = og_img.shape[:2]
    # height (h) and width (w) is incorrectly swapped in other places
    og_h = og_img_width
    h = og_h // 2
    og_w = og_img_height
    w = og_w // 2
    scale = 100
    roll = 0
    pitch = 0
    yaw = 0
    if isinstance(og_img, bool):
        print('Error during video loading')
        cv2.destroyWindow(scene_name)
        exit(0)

    cv2.namedWindow(scene_name)
    # + 1000 makes it possible to add coordinates outside image
    # this was possible with hardcoded max values if image size was small
    cv2.createTrackbar('h', scene_name, h, og_h + 1000, vertical_track)
    cv2.createTrackbar('w', scene_name, w, og_w + 1000, horizontal_track)
    cv2.createTrackbar('scale', scene_name, scale, 1000, scale_track)
    cv2.createTrackbar('roll', scene_name, roll, 360, roll_rot_track)
    cv2.createTrackbar('pitch', scene_name, pitch, 360, pitch_rot_track)
    cv2.createTrackbar('yaw', scene_name, yaw, 360, yaw_rot_track)
    cv2.createButton('Save', save)

    placements_from_file = []
    if args.previous:
        with open(_get_result_file()) as f:
            placements_from_file = json.load(f)
            if (len(keyframes) + 1) != len(placements_from_file):
                raise ValueError('Previous placement file includes different number of keyframes')
            _use_placement_from_file()

    showimg()


    while True:
        if cv2.waitKey():
            cv2.destroyWindow(scene_name)
            break
        