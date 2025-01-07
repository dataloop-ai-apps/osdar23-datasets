import dtlpy as dl
import os
import json
import raillabel
import uuid
import shutil
from zipfile import ZipFile
from scipy.spatial.transform import Rotation
import math
import numpy as np
from io import BytesIO

from dtlpylidar.parsers.base_parser import LidarFileMappingParser


class FixTransformation:
    @staticmethod
    def rotate_system(theta_x=None, theta_y=None, theta_z=None, radians: bool = True):
        if radians is False:
            theta_x = math.radians(theta_x) if theta_x else None
            theta_y = math.radians(theta_y) if theta_y else None
            theta_z = math.radians(theta_z) if theta_z else None

        rotation = np.identity(4)
        if theta_x is not None:
            rotation_x = np.array([
                [1, 0, 0, 0],
                [0, math.cos(theta_x), -math.sin(theta_x), 0],
                [0, math.sin(theta_x), math.cos(theta_x), 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_x
        if theta_y is not None:
            rotation_y = np.array([
                [math.cos(theta_y), 0, math.sin(theta_y), 0],
                [0, 1, 0, 0],
                [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_y
        if theta_z is not None:
            rotation_z = np.array([
                [math.cos(theta_z), -math.sin(theta_z), 0, 0],
                [math.sin(theta_z), math.cos(theta_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_z
        rotation[np.abs(rotation) < 1e-5] = 0
        return rotation

    @staticmethod
    def fix_camera_transformation(quaternion: np.ndarray, position: np.ndarray):
        # Rotation
        rotation_matrix = np.identity(4)
        rotation_matrix[0:3, 0:3] = Rotation.from_quat(quaternion).as_matrix()

        # Apply Rotation fix
        theta_y = 90
        theta_z = -90
        rotation_fix = FixTransformation.rotate_system(theta_y=theta_y, theta_z=theta_z, radians=False)
        rotation_matrix = rotation_matrix @ rotation_fix

        # Translation
        translation_matrix = np.identity(4)
        translation_matrix[0:3, 3] = position.tolist()

        # Extrinsic Matrix
        extrinsic_matrix = translation_matrix @ rotation_matrix
        translation_array = extrinsic_matrix[0:3, 3]
        translation = {"x": translation_array[0], "y": translation_array[1], "z": translation_array[2]}
        rotation_array = Rotation.as_quat(Rotation.from_matrix(extrinsic_matrix[0:3, 0:3]))
        # rotation_array = FixTransformation.rotate_coordinates(rotation=quaternion)
        rotation = {"x": rotation_array[0], "y": rotation_array[1], "z": rotation_array[2], "w": rotation_array[3]}
        return translation, rotation


class LidarCustomParser(LidarFileMappingParser):
    def __init__(self,
                 enable_ir_cameras: str,
                 enable_rgb_cameras: str,
                 enable_rgb_highres_cameras: str):
        self.attributes_id_mapping_dict = None
        # Handle Cameras Options
        ir_cameras = ['ir_center', 'ir_left', 'ir_right']
        rgb_cameras = ['rgb_center', 'rgb_left', 'rgb_right']
        rgb_highres_cameras = ['rgb_highres_center', 'rgb_highres_left', 'rgb_highres_right']

        self.camera_list = list()
        if str(enable_ir_cameras) == "true" or str(enable_ir_cameras) == "True":
            self.camera_list += ir_cameras
        if str(enable_rgb_cameras) == "true" or str(enable_rgb_cameras) == "True":
            self.camera_list += rgb_cameras
        if str(enable_rgb_highres_cameras) == "true" or str(enable_rgb_highres_cameras) == "True":
            self.camera_list += rgb_highres_cameras

        super().__init__()

    def attributes_id_mapping(self, dataset):
        recipe = dataset.recipes.list()[0]
        attributes_mapping = {}

        instructions = recipe.metadata.get('system', dict()).get('script', dict()).get('entryPoints', dict()).get(
            'annotation:context:set', dict()).get('_instructions', list())

        for instruction in instructions:
            instructions2 = instruction.get('body', dict()).get('block', dict()).get('_instructions', list())
            for instruction2 in instructions2:
                title = instruction2.get('title', None)
                key = instruction2.get('body', dict()).get('key', None)
                attributes_mapping[title] = key
        self.attributes_id_mapping_dict = attributes_mapping

    @staticmethod
    def extract_zip_file(zip_filepath: str):
        data_path = str(uuid.uuid4())

        try:
            os.makedirs(name=data_path, exist_ok=True)

            with ZipFile(zip_filepath, 'r') as zip_object:
                zip_object.extractall(path=os.path.join(".", data_path))

        except Exception as e:
            shutil.rmtree(path=data_path, ignore_errors=True)
            raise dl.exceptions.BadRequest(
                status_code="400",
                message=f"Failed due to the following error: {e}"
            )
        data_path = os.path.join(os.getcwd(), data_path)
        return data_path

    def upload_pcds_and_images(self, data_path: str, dataset: dl.Dataset, progress: dl.Progress = None):
        scene = None
        dir_items = os.listdir(path=data_path)
        for dir_item in dir_items:
            if ".json" in dir_item:
                try:
                    calibration_json = os.path.join(data_path, dir_item)
                    scene = raillabel.load(calibration_json)
                    break
                except:
                    continue

        if scene is None:
            dl.exceptions.NotFound("Couldn't find supported json for 'raillabel'")

        # Loop through frames
        frames = scene.frames
        total_lidar_frames = len(frames)
        modulo_report = total_lidar_frames // 10
        for lidar_frame, (frame_num, frame) in enumerate(frames.items()):
            # Sensor ego pose
            ego_pose = frame.sensors['lidar']
            pcd_filepath = os.path.join(data_path, ego_pose.uri[1:])
            dataset.items.upload(
                local_path=pcd_filepath,
                remote_path=f"/lidar",
                remote_name=f"{lidar_frame}.pcd",
                overwrite=True
            )
            print(
                f"Uploaded to 'lidar' directory, the file: '{pcd_filepath}', "
                f"as: '/lidar/{lidar_frame}.pcd'"
            )

            # Loop through images
            for idx, image in enumerate(self.camera_list):
                # Get sensor
                sensor_reference = frame.sensors[image]
                image_filepath = os.path.join(data_path, sensor_reference.uri[1:])
                ext = os.path.splitext(p=sensor_reference.uri)[1]
                dataset.items.upload(
                    local_path=image_filepath,
                    remote_path=f"/frames/{lidar_frame}",
                    remote_name=f"{idx}{ext}",
                    overwrite=True
                )
                print(
                    f"Uploaded to 'frames' directory, the file: '{image_filepath}', "
                    f"as: '/frames/{lidar_frame}/{idx}{ext}'"
                )

            if progress is not None:
                if lidar_frame % modulo_report == 0:
                    _progress = int(80 * ((lidar_frame + 1) / total_lidar_frames))
                    progress.update(progress=_progress, message="Uploading source data...")

    def create_mapping_json(self, data_path: str, dataset: dl.Dataset):
        scene = None
        dir_items = os.listdir(path=data_path)
        for dir_item in dir_items:
            if ".json" in dir_item:
                try:
                    calibration_json = os.path.join(data_path, dir_item)
                    scene = raillabel.load(calibration_json)
                    break
                except:
                    continue

        if scene is None:
            dl.exceptions.NotFound("Couldn't find supported json for 'raillabel'")

        output_frames = dict()

        # Loop through frames
        frames = scene.frames
        for lidar_frame, (frame_num, frame) in enumerate(frames.items()):
            # Sensor ego pose
            ego_pose = frame.sensors['lidar']

            # Output frame dict from `Metadata`
            output_frame_dict = {
                "metadata": {
                    "frame": int(frame_num),
                    "sensor_uri": ego_pose.uri
                },
                "path": f"lidar/{lidar_frame}.pcd",
                "timestamp": float(frame.timestamp),
                "position": {
                    "x": ego_pose.sensor.extrinsics.pos.x,
                    "y": ego_pose.sensor.extrinsics.pos.y,
                    "z": ego_pose.sensor.extrinsics.pos.z
                },
                "heading": {
                    "x": ego_pose.sensor.extrinsics.quat.x,
                    "y": ego_pose.sensor.extrinsics.quat.y,
                    "z": ego_pose.sensor.extrinsics.quat.z,
                    "w": ego_pose.sensor.extrinsics.quat.w
                },
                "images": dict()
            }

            # Loop through images
            for idx, image in enumerate(self.camera_list):
                # Get sensor
                sensor_reference = frame.sensors[image]

                # Get extrinsics
                extrinsics = sensor_reference.sensor.extrinsics
                quaternion = np.array([extrinsics.quat.x, extrinsics.quat.y, extrinsics.quat.z, extrinsics.quat.w])
                position = np.array([extrinsics.pos.x, extrinsics.pos.y, extrinsics.pos.z])

                # Apply camera transformation fix
                translation, rotation = FixTransformation.fix_camera_transformation(
                    quaternion=quaternion,
                    position=position
                )

                # Output image dict
                ext = os.path.splitext(p=sensor_reference.uri)[1]
                image_dict = {
                    "metadata": {
                        "frame": int(frame_num),
                        "image_uri": sensor_reference.uri,
                    },
                    "image_path": f"frames/{lidar_frame}/{idx}{ext}",
                    "timestamp": float(sensor_reference.timestamp),
                    "intrinsics": {
                        "fx": sensor_reference.sensor.intrinsics.camera_matrix[0],
                        "fy": sensor_reference.sensor.intrinsics.camera_matrix[5],
                        "cx": sensor_reference.sensor.intrinsics.camera_matrix[2],
                        "cy": sensor_reference.sensor.intrinsics.camera_matrix[6],
                    },
                    "extrinsics": {
                        "translation": {
                            "x": translation["x"],
                            "y": translation["y"],
                            "z": translation["z"]
                        },
                        "rotation": {
                            "x": rotation["x"],
                            "y": rotation["y"],
                            "z": rotation["z"],
                            "w": rotation["w"]
                        },
                    },
                    "distortion": {
                        "k1": sensor_reference.sensor.intrinsics.distortion[0],
                        "k2": sensor_reference.sensor.intrinsics.distortion[1],
                        "k3": sensor_reference.sensor.intrinsics.distortion[2],
                        "p1": sensor_reference.sensor.intrinsics.distortion[3],
                        "p2": sensor_reference.sensor.intrinsics.distortion[4]
                    }
                }
                output_frame_dict['images'][str(idx)] = image_dict

            output_frames[str(lidar_frame)] = output_frame_dict

        mapping_data = {"frames": output_frames}
        mapping_filepath = os.path.join(data_path, "mapping.json")
        with open(mapping_filepath, "w") as f:
            json.dump(obj=mapping_data, fp=f, indent=4)

        mapping_item = dataset.items.upload(local_path=mapping_filepath, overwrite=True)
        return mapping_item

    @staticmethod
    def upload_sem_ref_items(frames_item: dl.Item, ref_items_dict, dl_annotations):
        for annotation_uid, annotation_data in ref_items_dict.items():
            buffer = BytesIO()
            buffer.write(json.dumps(annotation_data.get('ref_item_json')).encode())
            buffer.seek(0)
            buffer.name = f"{annotation_uid}.json"
            sem_ref_item = frames_item.dataset.items.upload(
                remote_path="/.dataloop/sem_ref",
                local_path=buffer,
                overwrite=True
            )
            annotation = annotation_data.get('ref_annotation')
            ann_def = {"type": "ref_semantic_3d",
                       "label": annotation_data.get('label'),
                       "coordinates": {
                           "interpolation": "none",
                           "mode": "overwrite",
                           "ref": f"{sem_ref_item.id}",
                           "refType": "id"
                       },
                       "metadata": {
                           "system": {
                               "attributes": annotation_data.get('attributes'),
                               "frame": int(min(annotation_data.get('ref_item_json').get('frames').keys())),
                               "endFrame": int(max(annotation_data.get('ref_item_json').get('frames').keys()))
                           },
                           "object_uid": annotation.object.uid,
                           "uid": annotation.uid
                       }}
            dl_annotations.append(ann_def)

    def upload_pre_annotation_lidar(self, frames_item: dl.Item, data_path: str):
        self.attributes_id_mapping(dataset=frames_item.dataset)
        scene = None
        dir_items = os.listdir(path=data_path)
        for dir_item in dir_items:
            if ".json" in dir_item:
                try:
                    calibration_json = os.path.join(data_path, dir_item)
                    scene = raillabel.load(calibration_json)
                    break
                except:
                    continue
        dl_annotations = list()
        builder = frames_item.annotations.builder()

        next_object_id = 0
        object_id_map = dict()

        # Loop through frames
        frames = scene.frames
        ref_items_dict = dict()
        for lidar_frame, (frame_num, frame) in enumerate(frames.items()):
            print(f"Frame: {lidar_frame}")
            annotations = frame.annotations
            for annotation_id, annotation in annotations.items():
                label = annotation.object.type.replace('_', ' ')
                attributes = dict()
                for key, value in annotation.attributes.items():
                    if isinstance(value, bool) or key == 'carrying':
                        attributes[self.attributes_id_mapping_dict.get(key)] = value
                    else:
                        attributes[self.attributes_id_mapping_dict.get(key)] = str(value).replace(' %', '%')
                if isinstance(annotation, raillabel.format.Cuboid):
                    position = [
                        annotation.pos.x,
                        annotation.pos.y,
                        annotation.pos.z
                    ]
                    scale = [
                        annotation.size.x,
                        annotation.size.y,
                        annotation.size.z
                    ]
                    rotation = Rotation.from_quat([
                        annotation.quat.x,
                        annotation.quat.y,
                        annotation.quat.z,
                        annotation.quat.w
                    ])

                    annotation_definition = dl.Cube3d(
                        label=label,
                        position=position,
                        scale=scale,
                        rotation=rotation.as_euler(seq="xyz", degrees=False),
                        attributes=attributes
                    )

                    object_id = object_id_map.get(annotation.object.uid, None)

                    # Create new annotation
                    if object_id is None:
                        object_id = next_object_id
                        object_id_map[annotation.object.uid] = object_id
                        next_object_id += 1

                        metadata = {"object_uid": annotation.object.uid,
                                    "system": {"frameNumberBased": True},
                                    "uid": annotation.uid}
                        builder.add(
                            annotation_definition=annotation_definition,
                            frame_num=lidar_frame,
                            end_frame_num=lidar_frame,
                            object_id=str(object_id),
                            metadata=metadata
                        )
                    # Add frame for the existing annotation
                    else:
                        for idx, builder_annotation in enumerate(builder):
                            if builder_annotation.object_id == str(object_id):
                                builder[idx].add_frame(
                                    annotation_definition=annotation_definition,
                                    frame_num=lidar_frame,
                                )
                                builder[idx].end_frame = lidar_frame
                                builder[idx].end_time = lidar_frame
                                break

                    print(
                        f"Adding annotation: "
                        f"(Type: Cube3d, Label: {label}, ObjectID: {object_id}, Frame: {lidar_frame})"
                    )

                elif isinstance(annotation, raillabel.format.Poly3d):
                    ann_def = {
                        'type': 'polyline_3d',
                        'label': label,
                        'coordinates': {'interpolation': 'Linear',
                                        'lineType': 'linear',
                                        'points': [],
                                        'position': {}},
                        "metadata": {
                            "system": {
                                "frame": lidar_frame,
                                "attributes": attributes
                            },
                            "object_uid": f"{annotation.object.uid}",
                            "uid": annotation.uid
                        }
                    }
                    for point_3d in annotation.points:
                        dl_point = {'x': point_3d.x, 'y': point_3d.y, 'z': point_3d.z}
                        if len(ann_def['coordinates']['position']) == 0:
                            ann_def['coordinates']['position'] = dl_point
                        else:
                            ann_def['coordinates']['points'].append(dl_point)
                    dl_annotations.append(ann_def)
                elif isinstance(annotation, raillabel.format.Seg3d):
                    if annotation.uid not in ref_items_dict:
                        ref_items_dict[annotation.uid] = {
                            "ref_item_json": {
                                "type": "index",
                                "frames": {}
                            },
                            "label": label,
                            "attributes": attributes,
                            "ref_annotation": annotation}
                    ref_items_dict[annotation.uid]['ref_item_json']["frames"][str(lidar_frame)] = annotation.point_ids
        self.upload_sem_ref_items(frames_item=frames_item,
                                  ref_items_dict=ref_items_dict,
                                  dl_annotations=dl_annotations)
        frames_item.annotations.upload(dl_annotations)
        print(f"Annotations Object UID Mapping: {object_id_map}")
        builder.upload()

    def upload_pre_annotation_images(self, frames_item: dl.Item, data_path: str):
        if self.attributes_id_mapping_dict is None:
            self.attributes_id_mapping(dataset=frames_item.dataset)
        buffer = frames_item.download(save_locally=False)
        frames_item_data = json.load(buffer)
        scene = None
        dir_items = os.listdir(path=data_path)
        for dir_item in dir_items:
            if ".json" in dir_item:
                try:
                    calibration_json = os.path.join(data_path, dir_item)
                    scene = raillabel.load(calibration_json)
                    break
                except:
                    continue
        frames = scene.frames
        images_dict = dict()
        for frame_num, frame in enumerate(frames_item_data.get('frames')):
            images = frame.get('images', list())
            for image_num, image in enumerate(images):
                image_item = frames_item.dataset.items.get(item_id=image.get('image_id'))
                if frame_num not in images_dict:
                    images_dict[frame_num] = dict()
                images_dict[frame_num][image_num] = {
                    'builder': image_item.annotations.builder(),
                    'item': image_item
                }
        anno_count = 0
        for lidar_frame, (frame_num, frame) in enumerate(frames.items()):
            annotations = frame.annotations
            anno_count += len(annotations)
            for annotation_id, annotation in annotations.items():
                # # Option 1: IR images
                # if 'ir' not in annotation.sensor.uid:
                #     continue

                # # Option 2: RGB images
                # if not ('rgb' in annotation.sensor.uid and 'rgb_highres' not in annotation.sensor.uid):
                #     continue

                # Option 3: RGB Highres images
                if 'rgb_highres' not in annotation.sensor.uid:
                    continue

                label = annotation.object.type.replace('_', ' ')
                attributes = dict()
                metadata = {"object_uid": annotation.object.uid,
                            "uid": annotation.uid}
                for key, value in annotation.attributes.items():
                    if isinstance(value, bool) or key == 'carrying':
                        attributes[self.attributes_id_mapping_dict.get(key)] = value
                    else:
                        attributes[self.attributes_id_mapping_dict.get(key)] = str(value).replace(' %', '%')
                if isinstance(annotation, raillabel.format.Bbox):
                    img_num = 0 if 'center' in annotation.name else 1 if 'left' in annotation.name else 2
                    builder = images_dict[lidar_frame][img_num]['builder']
                    left = annotation.pos.x - annotation.size.x / 2
                    top = annotation.pos.y - annotation.size.y / 2
                    right = annotation.pos.x + annotation.size.x / 2
                    bottom = annotation.pos.y + annotation.size.y / 2
                    builder.add(annotation_definition=dl.Box(left=left,
                                                             right=right,
                                                             top=top,
                                                             bottom=bottom,
                                                             label=label,
                                                             attributes=attributes),
                                metadata=metadata)
                if isinstance(annotation, raillabel.format.Poly2d):
                    img_num = 0 if 'center' in annotation.name else 1 if 'left' in annotation.name else 2
                    builder = images_dict[lidar_frame][img_num]['builder']
                    coordinates = list()
                    for point in annotation.points:
                        coordinates.append({'x': point.x, 'y': point.y})
                    polyline_geo = dl.Polyline.from_coordinates(coordinates=coordinates)
                    builder.add(annotation_definition=dl.Polyline(geo=polyline_geo,
                                                                  label=label,
                                                                  attributes=attributes),
                                metadata=metadata)
        for frame_num, images in images_dict.items():
            for image_num, image in images.items():
                if len(image['builder']) > 0:
                    filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION, use_defaults=False)
                    image['builder'].item.annotations.delete(filters=filters)
                    image['builder'].upload()

    def custom_parse_data(self, zip_filepath: str, lidar_dataset: dl.Dataset, progress: dl.Progress = None):
        data_path = self.extract_zip_file(zip_filepath=zip_filepath)

        try:
            self.upload_pcds_and_images(data_path=data_path, dataset=lidar_dataset, progress=progress)
            if progress is not None:
                progress.update(progress=80, message="Parsing source data...")
            mapping_item = self.create_mapping_json(data_path=data_path, dataset=lidar_dataset)
            frames_item = self.parse_data(mapping_item=mapping_item)
            if progress is not None:
                progress.update(progress=90, message="Uploading annotations...")
            self.upload_pre_annotation_lidar(frames_item=frames_item, data_path=data_path)
            self.upload_pre_annotation_images(frames_item=frames_item, data_path=data_path)
        finally:
            shutil.rmtree(path=data_path, ignore_errors=True)

        return frames_item


def main():
    cp = LidarCustomParser(
        enable_ir_cameras="false",
        enable_rgb_cameras="false",
        enable_rgb_highres_cameras="true"
    )

    data_path = "./data"
    dataset = dl.datasets.get(dataset_id="66099e6289c8593e33498ce1")

    # cp.upload_pcds_and_images(data_path=data_path, dataset=dataset)
    mapping_item = cp.create_mapping_json(data_path=data_path, dataset=dataset)
    frames_item = cp.parse_data(mapping_item=mapping_item)
    # mapping_item = dataset.items.get(item_id="65f32fa45c63d275df8dc81d")

    # frames_item = dataset.items.get(filepath="/frames.json")
    cp.upload_pre_annotation_lidar(frames_item=frames_item, data_path=data_path)
    # cp.upload_pre_annotation_images(frames_item=frames_item, data_path=data_path)


if __name__ == "__main__":
    main()
