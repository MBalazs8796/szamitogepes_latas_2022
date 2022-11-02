import numpy as np
import math
import cv2
import json
from copy import deepcopy
from object_loader import OBJ

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
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        return image
    return False

def save(*args):
    global h, w, scale, roll, pitch, yaw
    with open('result.json', 'w') as fp:
        json.dump({
            'h' : h,
            'w' : w,
            'scale' : scale,
            'roll' : roll,
            'pitch' : pitch,
            'yaw' : yaw
        }, fp, indent=2)
    
    print('saved')

if __name__ == '__main__':

    scene_name = 'test_scene'


    obj = OBJ('lego.obj', swapyz=True)
    og_img = getFirstFrame('./vids/20221102_162217.mp4')
    
    rows, cols = og_img.shape[:2]
    h = 1000
    w = 1000
    scale = 100
    roll = 0
    pitch = 0
    yaw = 0
    if isinstance(og_img, bool):
        print('Error during video loading')
        cv2.destroyWindow(scene_name)
        exit(0)

    cv2.namedWindow(scene_name)
    cv2.createTrackbar('h', scene_name, h, 2500, vertical_track)
    cv2.createTrackbar('w', scene_name, w, 3500, horizontal_track)
    cv2.createTrackbar('Scale', scene_name, scale, 1000, scale_track)
    cv2.createTrackbar('roll', scene_name, roll, 360, roll_rot_track)
    cv2.createTrackbar('pitch', scene_name, pitch, 360, pitch_rot_track)
    cv2.createTrackbar('yaw', scene_name, yaw, 360, yaw_rot_track)
    cv2.createButton('Save', save)

    showimg()

    while True:
        if cv2.waitKey():
            cv2.destroyWindow(scene_name)
            break
        