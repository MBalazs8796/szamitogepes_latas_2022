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
    DOCKER_BUILDKIT=1 docker compose build

    # ORBSLAM2 with nvidea gpu
    docker compose run --rm orbslam python3 driver.py -c <config file path inside orbslam_driver directory>
    # ORBSLAM2 without using a gpu
    docker compose run --rm orbslam-nogpu python3 driver.py -c <config file path inside orbslam_driver directory>
    # ORBSLAM3
    docker compose run --rm orbslam3 python3 driver.py -c <config file path inside orbslam_driver directory>
    ```

### ORBSLAM3

Additional config parameters for ORBSLAM3 compared to ORBSLAM2:

```yaml
File.version: "1.0"
Camera.type: "PinHole"
Camera.width: 1280
Camera.height: 720
```
