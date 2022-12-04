import argparse
import math
import cv2
import json
import glob
import numpy as np
import re
from scipy.spatial.transform import Rotation
from copy import deepcopy
from object_placer.object_loader import OBJ
from pathlib import Path

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


def read_trajectory(filename):
    f = open(filename, 'r')
    # f = open('desk2-AllTrajectory.txt', 'r')

    lines = f.readlines()
    f.close()

    has_pose = []
    matrix_list = []
    i = 0
    while i < len(lines)-1:
        if lines[i] in ['\n', '\r\n']:
            i += 1
            break

        row = re.sub(r'\[|\]|\n|,|;', '', lines[i]).split()
        i += 1

        if len(row) == 0:
            has_pose.append(False)
        else:
            has_pose.append(True)
            matrix = []
            matrix.append([float(v) for v in row])
            for _ in range(3):
                row = re.sub(r'\[|\]|\n|,|;', '', lines[i]).split()
                matrix.append([float(v) for v in row])
                i += 1
            assert(len(matrix) == 4)
            matrix_list.append(matrix)
    matrix_list = np.array(matrix_list)
    has_pose = np.array(has_pose)

    return has_pose, matrix_list

def close_to_any(a, floats, **kwargs):
  return np.any(np.isclose(a, floats, **kwargs))

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
        R = Rotation.from_quat(nums[4:]).as_matrix()
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

    if not hasattr(render, 'points'):
      with open(f'./orbslam_driver/extracted/{scene_name}/result.json', 'r') as fp:
        sm = json.load(fp)[0]

      R = degree2R(sm['roll'], sm['pitch'], sm['yaw'])
      R = np.hstack([R, [[sm['h']],[sm['w']],[sm['scale']]]])
      R = np.vstack([R, [0,0,0,1]])
      
      point_list = list()
      for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])

        points = np.dot(points, scale_matrix)

        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        #points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        #point_list.append(cv2.perspectiveTransform(points.reshape(-1, 1, 3), R))
        #point_list.append(points.reshape(-1, 1, 3))
        point_list.append(points)
      setattr(render, 'point_list', point_list)


    for points in render.point_list:

        #print(points)
        #print(projection)
        #print(points.reshape(-1, 1, 3).shape)
        #print("################################")
        dst = cv2.transform(points.reshape(-1, 1, 3), projection)
        #print(dst.shape)
        #dst = dst @ np.diag([0.1, 0.1, 1])
        
        #for point in points:
          
        #print(points.reshape(3,3))
        #print(dst)
        #exit(69)
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


def _bg_filename_sort_key(filename: str):
  basename = Path(filename).name
  basename_parts_without_extension = basename.split('.')[:-1]
  return float('.'.join(basename_parts_without_extension))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', metavar=None, help='The name of the video')
    parser.add_argument('-c', '--config', metavar=None, help='The name of the camera config')
    args = parser.parse_args()

    scene_name = args.video
    camera_config_name = args.config

    with open(f'./orbslam_driver/extracted/{scene_name}/result.json', 'r') as fp:
      sm = json.load(fp)[0]


    # read file
    bg_filenames = glob.glob(f'./orbslam_driver/extracted/{scene_name}/rgb/*.png')
    bg_filenames.sort(key=_bg_filename_sort_key)
    poses, has_pose, names = read_kerframe_trajectory(f'./orbslam_driver/extracted/{scene_name}/KeyFrameTrajectory.txt', f'./orbslam_driver/extracted/{scene_name}/rgb.txt')

    video_path = f'./vids/{scene_name}.mp4'
           
    
    video = cv2.VideoCapture(video_path)
    
    ret = True
    
    if not video.isOpened():
        print('Error opening video')
        exit(69)

              
    bg_filenames = np.array(bg_filenames)
    bg_filenames = bg_filenames[has_pose]
    
    # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # print(frame_width)
    # print(frame_height)
    
    # writer = cv2.VideoWriter(f'{scene_name}_match_move.avi', cv2.VideoWriter_fourcc(*'XVID'), 49, (frame_width, frame_height))
    # while video.isOpened():
    #   s,f = video.read()
    #   writer.write(f)
    #   if not s:
    #     break
    # writer.release()
    # exit(69)
    
    assert(len(poses) == len(bg_filenames))

    n_frames = len(poses)

    obj = OBJ('./object_placer/lego.obj', swapyz=True)
    # obj = OBJ('fox.obj', swapyz=True)

    # Camera Intrinsics
    if camera_config_name == 'config_mb':
      im_w, im_h = 1280, 720
      fps = 5
      K =  np.array([ [870.25319293,  0, 637.28771858],
                      [0, 866.48681104, 354.55971258],
                      [0,       0,   1]])
    elif camera_config_name == 'config_tumvi':
      im_w, im_h = 512, 512
      fps = 20
      K =  np.array([ [190.978477,  0, 254.931706],
                      [0, 190.973307, 256.897442],
                      [0,       0,   1]])
    else:
      raise ValueError(f'Camera config with name {camera_config_name} not exists')

    # Convert K [3,3] to [4,4]
    K = np.hstack([K, np.zeros([3,1])])
    K = np.vstack([K, [0,0,0,1]])

    result_imgs = []
    for i in range(n_frames):
      print('='*10)
      print('Frame: ', i)

      img = cv2.imread(bg_filenames[i])


      #t_model = [[sm['h']],[sm['w']],[-500]]
      #R_model = degree2R(sm['roll'], sm['pitch'], sm['yaw'])
      R_model = degree2R(roll=0, pitch=0, yaw=0)
      t_model = np.array([[0, 0, -2]]).T  # 14000
      Rt_model = np.hstack([R_model, t_model]) 
      Rt_model = np.vstack([Rt_model, [0,0,0,1]])

      #Rt_model = Rt_model @ R

      # Camera Pose
      Rt_cam = poses[i]
      #print(Rt_cam)
      #Rt_cam = cv2.invertAffineTransform(Rt_cam)
      Rt_cam = np.linalg.pinv(Rt_cam)
      
      #Rt_cam = np.linalg.pinv(Rt_cam)
      Rt = Rt_cam @ Rt_model
      #Rt = approx_rotation(Rt[:-1, :])
      #Rt = np.linalg.pinv(Rt)
      #print('Rt', Rt, end='\n\n')
      P = K @ Rt

      #P = Rt
      #print('P', P, end='\n\n')
      P = P[:-1, :]  # [4,4] -> [3,4]
      img = render(img, obj, P, h=1, w=1, color=False, scale=1)
      # img = render(img, obj, Rt_model, h=0, w=0, color=False)

      cv2.imshow(scene_name, img)
      if cv2.waitKey() == ord('q'):
       break
      result_imgs.append(img)
    writer = cv2.VideoWriter(f'{scene_name}_match_move.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (im_w, im_h))
    for i in result_imgs:
      writer.write(i)
    writer.release()
    cv2.destroyAllWindows()