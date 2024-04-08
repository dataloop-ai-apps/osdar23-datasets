import dtlpy as dl
import custom_converter as lidar


class DatasetLidarOSDAR(dl.BaseServiceRunner):
    def __init__(self):
        dl.use_attributes_2(state=True)

        self.enable_ir_cameras = "false"
        self.enable_rgb_cameras = "false"
        self.enable_rgb_highres_cameras = "true"

    def upload_dataset(self, dataset: dl.Dataset, source: str):
        # TODO: used download command to get the zip

        # if "zip" not in item.mimetype:
        #     raise dl.exceptions.BadRequest(
        #         status_code="400",
        #         message="Input Item mimetype isn't zip!"
        #     )

        item: dl.Item
        lidar_parser = lidar.LidarCustomParser(
            enable_ir_cameras=self.enable_ir_cameras,
            enable_rgb_cameras=self.enable_rgb_cameras,
            enable_rgb_highres_cameras=self.enable_rgb_highres_cameras
        )
        frames_item = lidar_parser.custom_parse_data(item=item, overwrite=False)
        return frames_item


if __name__ == '__main__':
    sr = DatasetLidarOSDAR()
