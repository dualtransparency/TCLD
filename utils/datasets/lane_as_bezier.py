import os
import torch
import torchvision
import json
import numpy as np
from PIL import Image
import cv2

from .builder import DATASETS
from ..curve_utils import BezierSampler, get_valid_points
# from tools.curve_fitting_tools.loader import load_scene_gt
from utils.curve_utils import BezierCurve


def load_json(path, keys_to_int=False):
  """Loads content of a JSON file.

  :param path: Path to the JSON file.
  :return: Content of the loaded JSON file.
  """
  # Keys to integers.
  def convert_keys_to_int(x):
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

  with open(path, 'r') as f:
    if keys_to_int:
      content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
    else:
      content = json.load(f)

  return content


def load_scene_gt(path):
  """Loads content of a JSON file with ground-truth annotations.

  See docs/bop_datasets_format.md for details.

  :param path: Path to the JSON file.
  :return: Dictionary with the loaded content.
  """
  scene_gt = load_json(path, keys_to_int=True)

  for im_id, im_gt in scene_gt.items():
    for gt in im_gt:
      if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = np.array(gt['cam_R_m2c'], float).reshape((3, 3))
      if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = np.array(gt['cam_t_m2c'], float).reshape((3, 1))
  return scene_gt

def resize_img(img, target_size=512):
    height, width = img.shape[:2]
    scale_factor = target_size / max(height, width)

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    pad_w = target_size - new_width
    pad_h = target_size - new_height

    img_resized = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    return scale_factor, img_resized


class _BezierLaneDataset(torchvision.datasets.VisionDataset):
    # BezierLaneNet dataset, includes binary seg labels
    keypoint_color = [0, 0, 0]

    def __init__(self, root, image_set='train', transforms=None, transform=None, target_transform=None,
                 order=3, num_sample_points=100, aux_segmentation=False):
        super().__init__(root, transforms, transform, target_transform)
        self.aux_segmentation = aux_segmentation
        self.bezier_sampler = BezierSampler(order=order, num_sample_points=num_sample_points)
        if image_set == 'valfast':
            raise NotImplementedError('valfast Not supported yet!')
        elif image_set == 'test' or image_set == 'val':  # Different format (without lane existence annotations)
            self.test = 2
        elif image_set == 'val_train':
            self.test = 3
        else:
            self.test = 0

        self.init_dataset(root)

        if image_set != 'valfast':
            self.bezier_labels = os.path.join(self.bezier_labels_dir, image_set + '_' + str(order) + '.json')
        elif image_set == 'valfast':
            raise ValueError

        self.image_set = image_set
        self.splits_dir = os.path.join(root, 'lists')
        self._init_all()

    def init_dataset(self, root):
        raise NotImplementedError

    def __getitem__(self, index):
        if 'liquid' not in self.image_dir:
            # Return x (input image) & y (mask image, i.e. pixel-wise supervision) & lane existence (a list),
            # if not just testing,
            # else just return input image.
            img = Image.open(self.images[index]).convert('RGB')
            if self.test >= 2:
                target = self.masks[index]
            else:
                if self.aux_segmentation:
                    target = {'keypoints': self.beziers[index],
                            'segmentation_mask': Image.open(self.masks[index])}
                else:
                    target = {'keypoints': self.beziers[index]}

            # Transforms
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            if self.test == 0:
                target = self._post_process(target)

            return img, target
        else:
            return self.__getitem_liquid__(index)
        
    def __getitem_liquid__(self, index):
        target_size = 512

        rgb_fn = self.images[index]
        mask_fn = self.masks[index]
        bbox = self.bboxes[index]
        bezier = self.beziers[index]
        mask_fn = mask_fn.replace('mask', 'mask_visib')
        mask_liquid_fn = mask_fn.replace('mask_visib', 'mask_liquid_surface')

        img_array = cv2.imread(rgb_fn)
        mask_array = cv2.imread(mask_fn, -1)
        mask_liquid_array = cv2.imread(mask_liquid_fn, -1)
        
        #1. 得到ROI
        ###############################################

        img_array = img_array[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
        mask_array = mask_array[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        mask_liquid_array = mask_liquid_array[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        # new_bezier = bezier - np.array([bbox[0], bbox[1]])

        #2. resize
        ###############################################
        scale_factor, img_array = resize_img(img_array, target_size=target_size)
        scale_factor, mask_array = resize_img(mask_array, target_size=target_size)
        scale_factor, mask_liquid_array = resize_img(mask_liquid_array, target_size=target_size)

        # new_bezier = new_bezier * scale_factor
        
        ###############################################

        img = Image.fromarray(img_array)
        mask = Image.fromarray(mask_array)
        mask_liquid = Image.fromarray(mask_liquid_array)

        if self.test >= 2:
            target = mask_fn
        else:
            if self.aux_segmentation:
                target = {'keypoints': bezier,
                        'segmentation_mask': mask,
                        'segmentation_mask_liquid': mask_liquid}
            else:
                target = {'keypoints': bezier}
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        if self.test == 0:
            target = self._post_process(target)
        
        return img, target


    def __len__(self):
        return len(self.images)

    def loader_bezier(self):
        results = []
        with open(self.bezier_labels, 'r') as f:
            results += [json.loads(x.strip()) for x in f.readlines()]
        beziers = []
        for lanes in results:
            temp_lane = []
            for lane in lanes['bezier_control_points']:
                temp_cps = []
                for i in range(0, len(lane), 2):
                    temp_cps.append([lane[i], lane[i + 1]])
                temp_lane.append(temp_cps)
            beziers.append(np.array(temp_lane, dtype=np.float32))
        return beziers

    def _init_all(self):
        if 'liquid' not in self.image_dir:
            # Got the lists from 4 datasets to be in the same format
            data_list = 'train.txt' if self.image_set == 'val_train' else self.image_set + '.txt'
            split_f = os.path.join(self.splits_dir, data_list)
            with open(split_f, "r") as f:
                contents = ['d' + x.strip() for x in f.readlines()]
            if self.test == 2:  # Test
                self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
                self.masks = [os.path.join(self.output_prefix, x + self.output_suffix) for x in contents]
            elif self.test == 3:  # Test
                self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
                self.masks = [os.path.join(self.output_prefix, x[:x.find(' ')] + self.output_suffix) for x in contents]
            elif self.test == 1:  # Val
                self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
                self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
            else:  # Train
                self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
                if self.aux_segmentation:
                    self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
                self.beziers = self.loader_bezier()
        else:
            # load liquid dataset here
            if self.image_set == "train":
                self._init_all_liquid()
            else:
            # for ffb6d eval
                self._init_all_liquid_ffb6d()

    def _init_all_liquid(self):
        if self.image_set == 'train':
            visib_threshold = 0.2
        else:
            visib_threshold = 0.1
        self.images = []
        self.masks = []
        self.beziers = []
        self.bboxes = []

        scene_gt_infos = {}
        scene_gts = {}
        for scene in os.listdir(os.path.join(self.image_dir, self.image_set)):
            current_dir = os.path.join(self.image_dir, self.image_set, scene)
            scene_gt_info_fn = os.path.join(current_dir, 'scene_gt_info.json')
            scene_gt_fn = os.path.join(current_dir, 'scene_gt_liquid.json')

            scene_gt_info = load_scene_gt(scene_gt_info_fn)
            scene_gt = load_scene_gt(scene_gt_fn)
            
            scene_gts.update({scene: scene_gt})
            scene_gt_infos.update({scene: scene_gt_info})

        results = []
        with open(self.bezier_labels, 'r') as f:
            results += [json.loads(x.strip()) for x in f.readlines()]
        
        for item in results:
            mask_fn = item['raw_file']

            bezier_control_points = item['bezier_control_points']
            temp_lane = []
            for lane in bezier_control_points:
                temp_cps = []
                for i in range(0, len(lane), 2):
                    temp_cps.append([lane[i], lane[i+1]])
                temp_lane.append(temp_cps)
            beziers = np.array(temp_lane, dtype=np.float32)
            rgb_fn = mask_fn.replace('mask', 'rgb').split('_')[0] + '.png'

            scene = mask_fn.split('/')[-3]
            img_id = int(mask_fn.split('/')[-1].split('_')[0])
            rank = int(mask_fn.split('/')[-1].split('_')[1].split('.')[0])

            if len(scene_gts[scene][img_id]) != len(scene_gt_infos[scene][img_id]):
                   continue
            visib_fract = scene_gt_infos[scene][img_id][rank]['visib_fract']
            px_count_visib = scene_gt_infos[scene][img_id][rank]['px_count_visib']
            if not(visib_fract > visib_threshold and px_count_visib >= 1600):
                continue
            gt_info = scene_gt_infos[scene][img_id][rank]
            gt = scene_gts[scene][img_id][rank]
            # bbox = gt_info['bbox_visib']
            bbox = gt_info['bbox_obj']


            self.images.append(rgb_fn)
            self.masks.append(mask_fn)
            self.beziers.append(beziers)
            self.bboxes.append(bbox)
        print(f'Load all samples = {len(self.images)}')

    
    def _init_all_liquid_ffb6d(self):
        assert self.image_set == 'test' or self.image_set == 'val'
        
        self.images = []
        self.masks = []
        self.beziers = []
        self.bboxes = []

        supported_objects = ['175ml_flask', '25ml_flask', '75ml_flask', 'Grex']
        objid2object = {15:'25ml_flask', 19:'Grex', 16:'75ml_flask', 17:'175ml_flask'}

        scene_gt_infos = {}
        scene_gts = {}
        for scene in os.listdir(os.path.join(self.image_dir, self.image_set)):
            current_dir = os.path.join(self.image_dir, self.image_set, scene)
            scene_gt_info_fn = os.path.join(current_dir, 'scene_gt_info.json')
            scene_gt_fn = os.path.join(current_dir, 'scene_gt_liquid.json')

            scene_gt_info = load_scene_gt(scene_gt_info_fn)
            scene_gt = load_scene_gt(scene_gt_fn)
            
            scene_gts.update({scene: scene_gt})
            scene_gt_infos.update({scene: scene_gt_info})

        gts_ffb = {}
        for object in supported_objects:
            gt_ffb_fn = os.path.join(self.image_dir, object+'.json')
            with open(gt_ffb_fn, 'r') as f:
                gt = json.load(f)
            gts_ffb.update({object: gt})

        results = []
        with open(self.bezier_labels, 'r') as f:
            results += [json.loads(x.strip()) for x in f.readlines()]
        
        for item in results:
            mask_fn = item['raw_file']

            bezier_control_points = item['bezier_control_points']
            temp_lane = []
            for lane in bezier_control_points:
                temp_cps = []
                for i in range(0, len(lane), 2):
                    temp_cps.append([lane[i], lane[i+1]])
                temp_lane.append(temp_cps)
            beziers = np.array(temp_lane, dtype=np.float32)
            rgb_fn = mask_fn.replace('mask', 'rgb').split('_')[0] + '.png'

            scene = mask_fn.split('/')[-3]
            img_id = int(mask_fn.split('/')[-1].split('_')[0])
            rank = int(mask_fn.split('/')[-1].split('_')[1].split('.')[0])

            if len(scene_gts[scene][img_id]) != len(scene_gt_infos[scene][img_id]):
                   continue
            obj_id = scene_gts[scene][img_id][rank]['obj_id']
            if objid2object[obj_id] not in gts_ffb:
                continue

            if rgb_fn not in gts_ffb[objid2object[obj_id]]:
                continue

            gt_ffb = gts_ffb[objid2object[obj_id]][rgb_fn]
            bbox = gt_ffb['obj_bb']
            iou = gt_ffb['iou']
            if iou < 0.9:
                continue

            if bbox[2] < 50 or bbox[3] < 50:
                continue

            # visib_fract = scene_gt_infos[scene][img_id][rank]['visib_fract']
            # px_count_visib = scene_gt_infos[scene][img_id][rank]['px_count_visib']
            # # if not(visib_fract > visib_threshold and px_count_visib >= 1600):
            # #     continue
            # gt_info = scene_gt_infos[scene][img_id][rank]
            # gt = scene_gts[scene][img_id][rank]
            # # bbox = gt_info['bbox_visib']
            # bbox = gt_info['bbox_obj']


            self.images.append(rgb_fn)
            self.masks.append(mask_fn)
            self.beziers.append(beziers)
            self.bboxes.append(bbox)
        print(f'Load all samples = {len(self.images)}')

                

    def _post_process(self, target, ignore_seg_index=255):
        # Get sample points and delete invalid lines (< 2 points)
        if target['keypoints'].numel() != 0:  # No-lane cases can be handled in loss computation
            sample_points = self.bezier_sampler.get_sample_points(target['keypoints'])
            valid_lanes = get_valid_points(sample_points).sum(dim=-1) >= 2
            target['keypoints'] = target['keypoints'][valid_lanes]
            target['sample_points'] = sample_points[valid_lanes]
        else:
            target['sample_points'] = torch.tensor([], dtype=target['keypoints'].dtype)

        if 'segmentation_mask' in target.keys():  # Map to binary (0 1 255)
            # positive_mask = (target['segmentation_mask'] > 0) * (target['segmentation_mask'] != ignore_seg_index)
            positive_mask = (target['segmentation_mask'] > 0)
            target['segmentation_mask'][positive_mask] = 1
        if 'segmentation_mask_liquid' in target.keys():
            positive_mask = (target['segmentation_mask_liquid'] > 0)
            target['segmentation_mask_liquid'][positive_mask] = 1

        return target
    

# Liquid
@DATASETS.register()
class LiquidAsBezier(_BezierLaneDataset):
    colors = [
        [0, 0, 0],  # background
        [255, 0, 0], 
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        self.image_dir = root
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.output_prefix = './liquid_output'

# TuSimple
@DATASETS.register()
class TuSimpleAsBezier(_BezierLaneDataset):
    colors = [
        [0, 0, 0],  # background
        [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 255, 255],
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        self.image_dir = os.path.join(root, 'clips')
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'segGT6')
        self.output_prefix = 'clips'
        self.output_suffix = '.jpg'
        self.image_suffix = '.jpg'


# CULane
@DATASETS.register()
class CULaneAsBezier(_BezierLaneDataset):
    colors = [
        [0, 0, 0],  # background
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0],
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        self.image_dir = root
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'laneseg_label_w16')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = '.jpg'
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)


# LLAMAS
@DATASETS.register()
class LLAMAS_AsBezier(_BezierLaneDataset):
    colors = [
        [0, 0, 0],  # background
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0],
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        self.image_dir = os.path.join(root, 'color_images')
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'laneseg_labels')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = '.png'
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)


# Curvelanes
@DATASETS.register()
class Curvelanes_AsBezier(CULaneAsBezier):
    # TODO: Match formats
    colors = []

    def _init_all(self):
        # Got the lists from 4 datasets to be in the same format
        data_list = 'train.txt' if self.image_set == 'val_train' else self.image_set + '.txt'
        split_f = os.path.join(self.splits_dir, data_list)
        with open(split_f, "r") as f:
            contents = [x.strip() for x in f.readlines()]
        if self.test == 2:  # Test
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x + self.output_suffix) for x in contents]
        elif self.test == 3:  # Test
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x[:x.find(' ')] + self.output_suffix) for x in contents]
        elif self.test == 1:  # Val
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
        else:  # Train
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            if self.aux_segmentation:
                self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
            self.beziers = self.loader_bezier()
