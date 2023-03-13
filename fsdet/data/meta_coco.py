import contextlib
import io
import os

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fsdet.utils.file_io import PathManager
from pycocotools.coco import COCO
import pdb

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


__all__ = ["register_meta_coco"]



COCO_NOVEL_IDS = [
    1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72,
]

COCO_NOVEL_THING = [
    "person", "bicycle", "car", "motorcycle", "airplane", 
    "bus", "train", "boat", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "bottle", 
    "chair", "couch", "potted plant", "dining table", "tv",
]


def load_coco_json(json_file, image_root, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    is_shots = "shot" in dataset_name
    if 'allnovel' in dataset_name:
        include_classnames = metadata['novel_classes']
        print('Logging names for {} novel classes: '.format(len(include_classnames)), include_classnames)
    else:
        include_classnames = metadata["thing_classes"]

    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "cocosplit")
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-2].split("shot")[0]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = dataset_name.split("_")[-1].split("shot")[0]
        for idx, cls in enumerate(include_classnames):
            json_file = os.path.join(
                split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls)
            )
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
    else:
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))
    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    if 'allnovel' in dataset_name or 'allbase' in dataset_name:
        assert len(id_map) == 80
        if 'allbase' in dataset_name:
            id_map = {thing_id:cont_id for thing_id,cont_id in id_map.items() if thing_id not in COCO_NOVEL_IDS}

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]

    if is_shots:
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record["file_name"] = os.path.join(
                        image_root, img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
        new_dataset_dicts = {}
        for item in dataset_dicts:
            if item['image_id'] not in new_dataset_dicts:
                new_dataset_dicts[item['image_id']] = item
            else:
                new_dataset_dicts[item['image_id']]['annotations'].extend(item['annotations'])
        dataset_dicts = []
        for _,item in new_dataset_dicts.items():
            dataset_dicts.append(item)
    else:
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def register_meta_coco(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(annofile, imgdir, metadata, name),
    )
    
    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]
    # elif "_allbase" in name or "_allnovel" in name:
    #     base_cont_id = metadata['base_dataset_id_to_contiguous_id']
    #     novel_cont_id = {thing_id:cont_id+60 for thing_id,cont_id in metadata['novel_dataset_id_to_contiguous_id'].items()}
    #     metadata["thing_dataset_id_to_contiguous_id"].update(base_cont_id)
    #     metadata["thing_dataset_id_to_contiguous_id"].update(novel_cont_id)
    #     metadata["thing_classes"] = metadata['base_classes'] + metadata['novel_classes']

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )
