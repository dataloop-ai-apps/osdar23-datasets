import dtlpy as dl
import requests
import os
import logging

import custom_converter as lidar

logger = logging.getLogger()


class DatasetLidarOSDAR(dl.BaseServiceRunner):
    def __init__(self):
        dl.use_attributes_2(state=True)

        # self.dataset_url = "https://download.data.fid-move.de/dzsf/osdar23/1_calibration_1.2.zip"
        self.dataset_url = "https://download.data.fid-move.de/dzsf/osdar23/9_station_ruebenkamp_9.7.zip"
        self.zip_filename = "data.zip"

        self.enable_ir_cameras = "false"
        self.enable_rgb_cameras = "false"
        self.enable_rgb_highres_cameras = "true"

    def download_zip(self):
        # Download the file
        response = requests.get(url=self.dataset_url)
        zip_filepath = os.path.join(os.getcwd(), self.zip_filename)
        with open(zip_filepath, 'wb') as file:
            file.write(response.content)

        logger.info(msg=f"File downloaded to: {zip_filepath}")
        return zip_filepath

    def upload_dataset(self, dataset: dl.Dataset, source: str):
        if self.zip_filename not in os.listdir(path=os.getcwd()):
            zip_filepath = self.download_zip()
        else:
            zip_filepath = os.path.join(os.getcwd(), self.zip_filename)

        item: dl.Item
        lidar_parser = lidar.LidarCustomParser(
            enable_ir_cameras=self.enable_ir_cameras,
            enable_rgb_cameras=self.enable_rgb_cameras,
            enable_rgb_highres_cameras=self.enable_rgb_highres_cameras
        )
        frames_item = lidar_parser.custom_parse_data(item=item, overwrite=False)
        return frames_item


def test_download():
    sr = DatasetLidarOSDAR()
    sr.download_zip()


def main():
    test_download()


if __name__ == '__main__':
    main()
