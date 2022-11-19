import numpy as np
import math
import cv2
import json
import argparse
from copy import deepcopy
from object_loader import OBJ


def read_kerframe_trajectory(trajectory_fn, timestamp_fn):
    with open(trajectory_fn, 'r') as fp:  # 'KeyFrameTrajectory.txt'
      lines = fp.readlines()

    with open(timestamp_fn, 'r') as fp:
      time_lines = fp.readlines()

    Rt_list = []
    timestamp_list = []
    for line in lines:
      if line.strip():
        nums = [float(n) for n in line.split(' ')]
        timestamp = nums[0]
        t = np.array(nums[1:4]).reshape([3,1])
        R = quat2mat(nums[4:])
        Rt = np.hstack([R, t])
        Rt = np.vstack([Rt, [0,0,0,1]])
        Rt_list.append(Rt)
        timestamp_list.append(round(float(timestamp), 2))


    has_pose = list()
    names = list()
    for line in time_lines:
      if line.strip() and line[0] != '#':
        tstamp, imgpth = line.split(' ')
        names.append(imgpth.split('/')[-1].strip())
        if round(float(tstamp), 2) in timestamp_list:
          has_pose.append(True)
        else:
          has_pose.append(False)


    Rt_list = np.array(Rt_list)
    has_pose = np.array(has_pose)

    return Rt_list, has_pose, names

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    '''
    _FLOAT_EPS = np.finfo(float).eps
    x, y, z, w = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

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

        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = cv2.hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)
    #print('Render Success')

    return img

def approx_rotation(Rt):
  """  Get legal rotation matrix """
  # rotate teapot 90 deg around x-axis so that z-axis is up
  Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])

  # set rotation to best approximation
  R = Rt[:,:3]
  U,S,V = np.linalg.svd(R)
  R = np.dot(U,V)
  R[0,:] = -R[0,:] # change sign of x-axis

  # set translation
  t = Rt[:,3].reshape(-1)

  # setup 4*4 model view matrix
  M = np.eye(4)
  M[:3,:3] = np.dot(R,Rx)
  M[:3,3] = t
  return M

def _get_placement(i: int|None = None):
    global index, placements
    i = index if i is None else i

    if placements[0] is None:
        _set_placement_from_current_values(0)
    # fill placements until index
    for idx in range(1, i+1):
        placement = placements[idx]
        if placement is None:
            prev_p = placements[idx-1]
            _set_placement(
                i,
                h=prev_p['h'],
                w=prev_p['w'],
                scale=prev_p['scale'],
                roll=prev_p['roll'],
                pitch=prev_p['pitch'],
                yaw=prev_p['yaw']
            )
    placement = placements[i]

    # this can cause multiple image updates depending on how many values changed
    cv2.setTrackbarPos('h', scene_name, placement['h'])
    cv2.setTrackbarPos('w', scene_name, placement['w'])
    cv2.setTrackbarPos('scale', scene_name, placement['scale'])
    cv2.setTrackbarPos('roll', scene_name, placement['roll'])
    cv2.setTrackbarPos('pitch', scene_name, placement['pitch'])
    cv2.setTrackbarPos('yaw', scene_name, placement['yaw'])


def _set_placement(index, h, w, scale, roll, pitch, yaw):
    global placements

    placements[index] = {
        'h' : h,
        'w' : w,
        'scale' : scale,
        'roll' : roll,
        'pitch' : pitch,
        'yaw' : yaw
    }


def _set_placement_from_current_values(i: int|None = None):
    global index, h, w, scale, roll, pitch, yaw
    i = index if i is None else i

    _set_placement(i, h, w, scale, roll, pitch, yaw)


def showimg():
    global h, w, scale, roll, pitch, yaw,scene_name, og_img, K, index, poses

    R = degree2R(roll, pitch, yaw)
    t = np.array([[h, w, -500]]).T
    Rt_model = np.hstack([R, t]) 
    Rt_model = np.vstack([Rt_model, [0,0,0,1]])
    Rt = poses[index] @ Rt_model
    Rt = approx_rotation(Rt[:-1, :])
    P = K @ Rt
    P = P[:-1, :]

    img = render(og_img, obj, P, h=1, w=1, color=False, scale = scale)
    cv2.imshow(scene_name, img)

def frame_track(i):
    global index, og_img

    if i != index:
        _set_placement_from_current_values()
    index = i
    og_img = keyframes[index]
    _get_placement()

    showimg()

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
    global placements

    if not next_image():
        with open(_get_result_file(), 'w') as f:
            json.dump(placements, f, indent=4)
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
    global index, scene_name, keyframes

    _set_placement_from_current_values()
    if index < (len(keyframes) - 1):
        cv2.setTrackbarPos('frame', scene_name, index + 1)
        return True
    return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', metavar=None, help='The name of the video')
    parser.add_argument('-p', '--previous', action='store_true', help='Use previous object placement file')
    parser.add_argument('-c', '--config', metavar=None, help='The name of the camera config')
    args = parser.parse_args()

    scene_name = args.video

    index = 0
    results = list()
    #og_img = getFirstFrame(args.video)
    keyframes = get_keyframes(args.video)
    og_img = keyframes[0]

    obj = OBJ('lego.obj', swapyz=True)

    if args.config == 'config_mb':
      im_w, im_h = 1280, 720
      K =  np.array([ [870.25319293,  0, 637.28771858],
                      [0, 866.48681104, 354.55971258],
                      [0,       0,   1]])

    K = np.hstack([K, np.zeros([3,1])])
    K = np.vstack([K, [0,0,0,1]])

    poses, _, _ = read_kerframe_trajectory(f'../orbslam_driver/extracted/{args.video}/KeyFrameTrajectory.txt', f'../orbslam_driver/extracted/{args.video}/rgb.txt')

    og_img_height, og_img_width = og_img.shape[:2]
    # height (h) and width (w) is incorrectly swapped in other places
    og_h = og_img_width
    h = 0
    og_w = og_img_height
    w = 0
    scale = 10
    roll = 0
    pitch = 0
    yaw = 0
    if isinstance(og_img, bool):
        print('Error during video loading')
        cv2.destroyWindow(scene_name)
        exit(0)

    placements = [None] * len(keyframes)
    if args.previous:
        with open(_get_result_file()) as f:
            placements = json.load(f)
            if len(keyframes) != len(placements):
                raise ValueError('Previous placement file includes different number of keyframes')

    cv2.namedWindow(scene_name)
    if len(keyframes) > 1:
        cv2.createTrackbar('frame', scene_name, index, len(keyframes) - 1, frame_track)
    else:
        frame_track(index)
    # + 1000 makes it possible to add coordinates outside image
    # this was possible with hardcoded max values if image size was small
    cv2.createTrackbar('h', scene_name, h, og_h + 1000, vertical_track)
    cv2.createTrackbar('w', scene_name, w, og_w + 1000, horizontal_track)
    cv2.createTrackbar('scale', scene_name, scale, 1000, scale_track)
    cv2.createTrackbar('roll', scene_name, roll, 360, roll_rot_track)
    cv2.createTrackbar('pitch', scene_name, pitch, 360, pitch_rot_track)
    cv2.createTrackbar('yaw', scene_name, yaw, 360, yaw_rot_track)
    #cv2.createButton('Save', save)

    _get_placement()


    while True:
        key = cv2.waitKey()
        # keyboard shortcut for next frame / save
        if key == ord('s'):
            save()
        # keyboard shortcut to load previous frame's placement to current frame
        elif key == ord('p'):
            if index > 0:
                _get_placement(index - 1)
        # keyboard shortcut to select previous frame
        elif key == ord('a'):
            cv2.setTrackbarPos('frame', scene_name, index - 1)
        # keyboard shortcut to select next frame
        elif key == ord('d'):
            cv2.setTrackbarPos('frame', scene_name, index + 1)
        else:
            cv2.destroyWindow(scene_name)
            break
