import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path


def _process_video(filename: str):
    """
    Converts video to GIF with FFmpeg

    see https://superuser.com/a/556031 for more information about used options
    """

    basename = Path(filename).name
    basename_parts_without_extension = basename.split('.')[:-1]
    basename_without_extension = '.'.join(basename_parts_without_extension)

    result_dir = "./result"
    Path(result_dir).mkdir(exist_ok=True)

    args = [
        "ffmpeg",
        "-i", filename,
        "-vf", f"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0", # infinite looping
        f"{result_dir}/{basename_without_extension}.gif"
    ]
    subprocess.run(args)


if __name__ == '__main__':
    parser = ArgumentParser(description="Converts videos to GIF with FFmpeg")
    args = parser.parse_args()

    for root, dirs, files in os.walk('./vids'):
        for file in files:
            video = os.path.join(root, file)
            _process_video(video)
