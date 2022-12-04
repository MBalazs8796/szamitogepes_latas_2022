import argparse

from match_mover_alt.match_mover import MatchMover

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--video',
        metavar=None,
        help='The name of the video'
    )
    parser.add_argument(
        '-c', '--config',
        metavar=None,
        choices=['config_az', 'config_fm', 'config_mb', 'config_tumvi'],
        help='The name of the camera config'
    )
    args = parser.parse_args()

    video_name = args.video
    camera_config_name = args.config

    match_mover = MatchMover(
        video_name=video_name,
        camera_config_name=camera_config_name
    )
    match_mover.create_video()
