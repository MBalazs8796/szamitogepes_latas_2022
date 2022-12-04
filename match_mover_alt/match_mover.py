import glob
from pathlib import Path

import cv2
import matplotlib.colors
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from object_placer.object_loader import OBJ


class MatchMover:
    def __init__(self, video_name: str, camera_config_name: str):
        self.video_name = video_name
        self.camera_config_name = camera_config_name

    def _video_folder(self):
        return f'./orbslam_driver/extracted/{self.video_name}'

    def create_video(self):
        bg_filenames = self.read_rgb_file()
        has_pose, poses = self.read_keyframe_trajectory_file()

        # remove filenames that do not have a pose
        bg_filenames = bg_filenames[has_pose]
        n_frames = len(poses)
        assert (n_frames == len(bg_filenames))

        obj = MatchMover.load_object()

        # Camera Intrinsics
        if self.camera_config_name == 'config_az':
            im_w, im_h = 1920, 1080
            fps = 30
            K = np.array(
                [
                    [1255.9, 0, 640.0],
                    [0, 1262.28, 360.0],
                    [0, 0, 1]
                ]
            )
        elif self.camera_config_name == 'config_fm':
            im_w, im_h = 1920, 1080
            fps = 30
            K = np.array(
                [
                    [1715.72378, 0, 936.080577],
                    [0, 1715.86524, 499.547070],
                    [0, 0, 1]
                ]
            )
        elif self.camera_config_name == 'config_mb':
            im_w, im_h = 1280, 720
            fps = 30
            K = np.array(
                [
                    [870.25319293, 0, 637.28771858],
                    [0, 866.48681104, 354.55971258],
                    [0, 0, 1]
                ]
            )
        elif self.camera_config_name == 'config_tumvi':
            im_w, im_h = 512, 512
            fps = 20
            K = np.array(
                [
                    [190.978477, 0, 254.931706],
                    [0, 190.973307, 256.897442],
                    [0, 0, 1]
                ]
            )
        else:
            raise ValueError('Unknown camera config')

        # Convert K [3,3] to [4,4]
        K = np.hstack([K, np.zeros([3, 1])])
        K = np.vstack([K, [0, 0, 0, 1]])

        video_writer = cv2.VideoWriter(
            f'{self.video_name}_match_move.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (im_w, im_h)
        )
        for i in tqdm(range(n_frames)):

            img = cv2.imread(bg_filenames[i])

            # Extrinsics
            t_model = np.array([[0, 0, -10]]).T
            R_model = MatchMover.degree2R(roll=240, pitch=0, yaw=0)
            Rt_model = np.hstack([R_model, t_model])  # [3,4]
            Rt_model = np.vstack([Rt_model, [0, 0, 0, 1]])  # [4,4]

            # Camera Pose
            Rt_cam = poses[i]

            Rt = Rt_cam @ Rt_model

            P = K @ Rt
            P = P[:-1, :]  # [4,4] -> [3,4]

            img = MatchMover.render(
                img, obj, P, h=0, w=0, color=False, scale=1)

            video_writer.write(img)
        video_writer.release()

    @staticmethod
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
        r = Rotation.from_quat(q)
        return r.as_matrix()

    @staticmethod
    def read_keyframe_trajectory(trajectory_filename, rgb_filename):
        lines: list[str] = []
        time_lines: list[str] = []
        with open(trajectory_filename, 'r') as f:
            lines = f.readlines()
        with open(rgb_filename, 'r') as f:
            time_lines = f.readlines()

        Rt_list = []
        timestamp_list = []
        for line in lines:
            if line.strip():
                nums = [float(n) for n in line.split(' ')]
                timestamp = nums[0]
                t = np.array(nums[1:4]).reshape([3, 1])
                R = MatchMover.quat2mat(nums[4:])
                # inverse of rotation matrix from quaternion
                R = np.linalg.pinv(R)
                Rt = np.hstack([R, t])
                Rt = np.vstack([Rt, [0, 0, 0, 1]])
                Rt_list.append(Rt)
                timestamp_list.append(timestamp)

        has_pose = []
        for line in time_lines:
            if line.strip() and line[0] != '#':
                tstamp, imgpth = line.split(' ')
                if float(tstamp) in timestamp_list:
                    has_pose.append(True)
                else:
                    has_pose.append(False)

        Rt_list = np.array(Rt_list)
        has_pose = np.array(has_pose)

        return has_pose, Rt_list

    def read_keyframe_trajectory_file(self):
        return MatchMover.read_keyframe_trajectory(
            f'{self._video_folder()}/KeyFrameTrajectory.txt',
            f'{self._video_folder()}/rgb.txt'
        )

    @staticmethod
    def degree2R(roll: int, pitch: int, yaw: int):
        r = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True)

        return r.as_matrix()

    @staticmethod
    def render(img, obj, projection, h, w, color=False, scale=5):
        """
        Render a loaded obj model into the current video frame
        Input:
        img: bg image
        obj: loaded 3d model
        projection: 4x4 transformation matrix
        """
        vertices = obj.vertices
        scale_matrix = np.eye(3) * scale
        # h, w = model.shape

        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1]
                              for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            # render model in the middle of the reference surface. To do so,
            # model points must be displaced
            points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]]
                              for p in points])
            dst = cv2.perspectiveTransform(
                points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
            else:
                color = matplotlib.colors.to_rgb(face[-1])
                color = color[::-1]  # reverse
                cv2.fillConvexPoly(img, imgpts, color)

        return img

    def read_rgb_file(self):
        def _bg_filename_sort_key(filename: str):
            basename = Path(filename).name
            basename_parts_without_extension = basename.split('.')[:-1]
            return float('.'.join(basename_parts_without_extension))

        filenames = glob.glob(
            f'./orbslam_driver/extracted/{self.video_name}/rgb/*.png'
        )
        filenames.sort(key=_bg_filename_sort_key)
        return np.array(filenames)

    @staticmethod
    def load_object():
        return OBJ('./object_placer/lego_old.obj', swapyz=True)
