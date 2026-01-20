# Use an appropriate base image
FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

# Install required Python packages
RUN pip install numba \
    shapely \
    scipy \
    raillabel \
    git+https://github.com/dataloop-ai-apps/dtlpy-lidar
