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
