import dtlpy as dl
import requests
import os
import logging
import json

import custom_converter as lidar

logger = logging.getLogger(name='osdar-dataset')


class DatasetLidarOSDAR(dl.BaseServiceRunner):
    def __init__(self):
        dl.use_attributes_2(state=True)

        # Original sources
        # self.dataset_url = "https://download.data.fid-move.de/dzsf/osdar23/1_calibration_1.1.zip"  # 10 Frame
        # self.dataset_url = "https://download.data.fid-move.de/dzsf/osdar23/1_calibration_1.2.zip"  # 100 Frames

        self.dataset_url = "https://storage.googleapis.com/model-mgmt-snapshots/datasets-OSDAR2023/1_calibration_1_1_subset.zip"
        self.zip_filename = "data.zip"
        self.ontology_filename = "osdar_ontology.json"

        self.enable_ir_cameras = "false"
        self.enable_rgb_cameras = "false"
        self.enable_rgb_highres_cameras = "true"

    def _import_recipe_ontology(self, dataset: dl.Dataset) -> dl.Recipe:
        recipe: dl.Recipe = dataset.recipes.list()[0]
        ontology: dl.Ontology = recipe.ontologies.list()[0]

        new_ontology_filepath = os.path.join(os.path.dirname(str(__file__)), self.ontology_filename)
        with open(file=new_ontology_filepath, mode='r') as file:
            new_ontology_json = json.load(fp=file)

        ontology.copy_from(ontology_json=new_ontology_json)
        return recipe

    def _download_zip(self, progress: dl.Progress = None) -> str:
        zip_filepath = os.path.join(os.getcwd(), self.zip_filename)
        chunk_size = 8192
        with requests.get(self.dataset_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('Content-Length', 0))
            total_chunks = (total_size // chunk_size) + (1 if total_size % chunk_size != 0 else 0)
            modulo_report = total_chunks // 10
            with open(zip_filepath, 'wb') as f:
                for i, chunk in enumerate(r.iter_content(chunk_size=chunk_size)):
                    f.write(chunk)
                    if progress is not None:
                        if i % modulo_report == 0:
                            _progress = int(40 * ((i + 1) / total_chunks))
                            progress.update(progress=_progress, message="Downloading dataset for source...")

        logger.info(msg=f"File downloaded to: {zip_filepath}")
        return zip_filepath

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress: dl.Progress = None) -> dl.Item:
        self._import_recipe_ontology(dataset=dataset)
        if self.zip_filename not in os.listdir(path=os.getcwd()):
            zip_filepath = self._download_zip(progress=progress)
        else:
            zip_filepath = os.path.join(os.getcwd(), self.zip_filename)

        item: dl.Item
        lidar_parser = lidar.LidarCustomParser(
            enable_ir_cameras=self.enable_ir_cameras,
            enable_rgb_cameras=self.enable_rgb_cameras,
            enable_rgb_highres_cameras=self.enable_rgb_highres_cameras
        )
        frames_item = lidar_parser.custom_parse_data(zip_filepath=zip_filepath, lidar_dataset=dataset,
                                                     progress=progress)
        return frames_item


def test_download():
    sr = DatasetLidarOSDAR()
    sr._download_zip()


def test_import_recipe_ontology():
    dataset_id = "66325a24241a71f884f78431"

    dataset = dl.datasets.get(dataset_id=dataset_id)
    sr = DatasetLidarOSDAR()
    sr._import_recipe_ontology(dataset=dataset)


def test_dataset_import():
    dataset_id = "663b93cfd03cf2f75ddeff4f"

    dataset = dl.datasets.get(dataset_id=dataset_id)
    sr = DatasetLidarOSDAR()
    sr.upload_dataset(dataset=dataset, source="")


def main():
    test_download()
    # test_import_recipe_ontology()
    # test_dataset_import()


if __name__ == '__main__':
    main()
