# orbslam_driver

## Requiremets

- Linux
- Docker
- Docker Compose (v2)

## Instructions

1. Create a folder named 'vids' next to the driver.py, and put the test videos inside the folder
2. Run the driver with Docker to use orbslam.

    ```sh
    xhost +
    docker compose build

    # with nvidea gpu
    docker compose run --rm orbslam python3 driver.py -c <config file path inside orbslam_driver directory>
    # without using a gpu
    docker compose run --rm orbslam-nogpu python3 driver.py -c <config file path inside orbslam_driver directory>
    ```

### ORBSLAM3

```sh
xhost +
DOCKER_BUILDKIT=1 docker compose build

docker compose run --rm orbslam3
/ORB_SLAM3/Examples/Monocular/mono_tum /ORB_SLAM3/Vocabulary/ORBvoc.txt /data/${config_name}.yaml /data/extracted/${video_name}/
cat /ORB_SLAM3/KeyFrameTrajectory.txt
```

Additional config parameters for ORBSLAM3 compared to ORBSLAM2:

```yaml
File.version: "1.0"
Camera.type: "PinHole"
Camera.width: 1280
Camera.height: 720
```
