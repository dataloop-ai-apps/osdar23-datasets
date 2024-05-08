from dtlpylidar.parser_base import extrinsic_calibrations
from dtlpylidar.parser_base import images_and_pcds, camera_calibrations, lidar_frame, lidar_scene
import os
import dtlpy as dl
import json
from io import BytesIO
import uuid
import logging
import shutil

logger = logging.Logger(name="file_mapping_parser")


class LidarFileMappingParser:
    def __init__(self):
        self.mapping_data = dict()
        self.dataset = None
        self.jsons_path = ""
        self.absolute_path_search = True

    def parse_lidar_data(self, mapping_item: dl.Item):
        scene = lidar_scene.LidarScene()
        frames = self.mapping_data.get("frames", dict())
        for frame_num, frame_details in frames.items():
            try:
                if self.absolute_path_search:
                    print(f"Search PCD {frame_num} in the absolute path")
                    pcd_filepath = os.path.join(self.jsons_path, frame_details.get("path"))
                    pcd_filepath = pcd_filepath.replace(".pcd", ".json")
                    with open(pcd_filepath, 'r') as f:
                        pcd_json = json.load(f)
                else:
                    raise dl.exceptions.NotFound(message="Trigger Exception")
            except:
                self.absolute_path_search = False
                print(f"Search PCD {frame_num} in the relative path")
                pcd_filepath = os.path.join(self.jsons_path, mapping_item.dir[1:], frame_details.get("path"))
                pcd_filepath = pcd_filepath.replace(".pcd", ".json")
                with open(pcd_filepath, 'r') as f:
                    pcd_json = json.load(f)
            try:
                ground_map_id = pcd_json["metadata"]["user"]["lidar_ground_detection"]["groundMapId"]
            except (KeyError, TypeError):
                ground_map_id = None

            pcd_translation = extrinsic_calibrations.Translation(
                x=frame_details.get("position", dict()).get("x", 0),
                y=frame_details.get("position", dict()).get("y", 0),
                z=frame_details.get("position", dict()).get("z", 0)
            )
            pcd_rotation = extrinsic_calibrations.QuaternionRotation(
                x=frame_details.get("heading", dict()).get("x", 0),
                y=frame_details.get("heading", dict()).get("y", 0),
                z=frame_details.get("heading", dict()).get("z", 0),
                w=frame_details.get("heading", dict()).get("w", 0)
            )
            pcd_time_stamp = frame_details.get("timestamp", "")

            scene_pcd_item = images_and_pcds.LidarPcdData(
                item_id=pcd_json.get("id"),
                ground_id=ground_map_id,
                remote_path=pcd_json.get("filename"),
                extrinsic=extrinsic_calibrations.Extrinsic(
                    rotation=pcd_rotation,
                    translation=pcd_translation
                ),
                timestamp=pcd_time_stamp
            )
            lidar_frame_images = list()
            frame_images = frame_details.get("images", list())
            for image_num, image_details in frame_images.items():
                try:
                    if self.absolute_path_search:
                        print(f"Search image {image_num} for frame {frame_num} in the absolute path")
                        image_filepath = os.path.join(self.jsons_path, image_details.get("image_path"))
                        image_filepath = image_filepath.replace(".png", ".json")
                        with open(image_filepath, 'r') as f:
                            image_json = json.load(f)
                    else:
                        raise dl.exceptions.NotFound(message="Trigger Exception")
                except:
                    self.absolute_path_search = False
                    print(f"Search image {image_num} for frame {frame_num} in the relative path")
                    image_filepath = os.path.join(self.jsons_path, mapping_item.dir[1:], image_details.get("image_path"))
                    image_filepath = image_filepath.replace(".png", ".json")
                    with open(image_filepath, 'r') as f:
                        image_json = json.load(f)

                camera_id = f"{image_num}_frame_{frame_num}"
                image_timestamp = image_details.get("timestamp")
                camera_translation = extrinsic_calibrations.Translation(
                    x=image_details.get("extrinsics", dict()).get("translation").get("x", 0),
                    y=image_details.get("extrinsics", dict()).get("translation").get("y", 0),
                    z=image_details.get("extrinsics", dict()).get("translation").get("z", 0)
                )
                camera_rotation = extrinsic_calibrations.QuaternionRotation(
                    x=image_details.get("extrinsics", dict()).get("rotation").get("x", 0),
                    y=image_details.get("extrinsics", dict()).get("rotation").get("y", 0),
                    z=image_details.get("extrinsics", dict()).get("rotation").get("z", 0),
                    w=image_details.get("extrinsics", dict()).get("rotation").get("w", 0)
                )
                camera_intrinsic = camera_calibrations.Intrinsic(
                    fx=image_details.get("intrinsics", dict()).get("fx", 0),
                    fy=image_details.get("intrinsics", dict()).get("fy", 0),
                    cx=image_details.get("intrinsics", dict()).get("cx", 0),
                    cy=image_details.get("intrinsics", dict()).get("cy", 0)
                )
                camera_distortion = camera_calibrations.Distortion(
                    k1=image_details.get("distortion", dict()).get("k1", 0),
                    k2=image_details.get("distortion", dict()).get("k2", 0),
                    k3=image_details.get("distortion", dict()).get("k3", 0),
                    p1=image_details.get("distortion", dict()).get("p1", 0),
                    p2=image_details.get("distortion", dict()).get("p2", 0)
                )

                lidar_camera = camera_calibrations.LidarCameraData(
                    cam_id=camera_id,
                    intrinsic=camera_intrinsic,
                    extrinsic=extrinsic_calibrations.Extrinsic(
                        rotation=camera_rotation,
                        translation=camera_translation
                    ),
                    channel=image_details.get("image_path"),
                    distortion=camera_distortion
                )

                scene.add_camera(lidar_camera)
                scene_image_item = images_and_pcds.LidarImageData(
                    item_id=image_json.get("id"),
                    lidar_camera=lidar_camera,
                    remote_path=image_json.get("filename"),
                    timestamp=image_timestamp
                )
                lidar_frame_images.append(scene_image_item)

            frame_item = lidar_frame.LidarSceneFrame(
                lidar_frame_pcd=scene_pcd_item,
                lidar_frame_images=lidar_frame_images
            )
            scene.add_frame(frame_item)
        buffer = BytesIO()
        buffer.write(json.dumps(scene.to_json(), default=lambda x: None).encode())
        buffer.seek(0)
        buffer.name = "frames.json"
        frames_item = self.dataset.items.upload(
            remote_path="{}".format(mapping_item.dir),
            local_path=buffer,
            overwrite=True,
            item_metadata={
                "system": {
                    "shebang": {
                        "dltype": "PCDFrames"
                    }
                },
                "fps": 1
            }
        )
        return frames_item

    def parse_data(self, mapping_item: dl.Item):
        if "json" not in mapping_item.metadata.get("system", dict()).get("mimetype"):
            raise Exception("Expected item of type json")

        buffer = mapping_item.download(save_locally=False)
        self.mapping_data = json.loads(buffer.getvalue())

        self.dataset = mapping_item.dataset
        uid = str(uuid.uuid4())
        base_dataset_name = self.dataset.name.replace(":", "-")
        base_path = "{}_{}".format(base_dataset_name, uid)
        try:
            items_download_path = os.path.join(os.getcwd(), base_path)
            self.dataset.download_annotations(local_path=items_download_path)
            self.jsons_path = os.path.join(items_download_path, "json")
            frames_item = self.parse_lidar_data(mapping_item=mapping_item)
        except Exception as e:
            raise dl.exceptions.BadRequest(
                status_code="400",
                message=f"Encountered the following error: {e}"
            )
        finally:
            shutil.rmtree(base_path, ignore_errors=True)
        return frames_item


if __name__ == '__main__':
    item_id = "65269bdab1061a007c78730c"

    parser = LidarFileMappingParser()
    item = dl.items.get(item_id=item_id)
    print(parser.parse_data(mapping_item=item))
