import os
import numpy as np
from tqdm import tqdm
import json
import cv2
from collections import OrderedDict
from importmagician import import_from
with import_from('./'):
    from utils.curve_utils import BezierCurve
from scipy.interpolate import InterpolatedUnivariateSpline
from liquid_eval import interp


def del_double(lanes):
    lanes = sorted(lanes, key=lambda p: p[0])
    new_lanes = []
    previous = lanes[0]
    for i in range(1, len(lanes)):
        # if lanes[i][0] != previous[0]:
        if abs(lanes[i][0] - previous[0]) >= 0.01:
          new_lanes.append(lanes[i])
          previous = lanes[i]
    return new_lanes


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


def process_lanes(obj_id, liquid_label):
    if obj_id == 19:
        lanes = []
        for key, value in liquid_label.items():
            lanes.append(value)
        lanes = sorted(lanes, key=lambda p: p[1], reverse=True)[:(len(lanes) // 2)]

    else:
        A = np.array(liquid_label['A']) 
        B = np.array(liquid_label['B']) 
        C = np.array(liquid_label['C']) 
        D = np.array(liquid_label['D'])

        lanes_length = np.array([np.linalg.norm(A-C), np.linalg.norm(A-D), np.linalg.norm(B-C), np.linalg.norm(B-D)])
        max_id = np.argmax(lanes_length)
        if max_id == 0:
            lanes = interpolate_points(A, C)
        elif max_id == 1:
            lanes = interpolate_points(A, D) 
        elif max_id == 2:
            lanes = interpolate_points(B, C) 
        elif max_id == 3:
            lanes = interpolate_points(B, D) 

    lanes = np.array(lanes)
    return lanes


def interpolate_points(start_point, end_point, num_points=20):
    step_x = (end_point[0] - start_point[0]) / (num_points - 1)

    points = []
    points.append([start_point[0], start_point[1]])
    for i in range(1, num_points - 1):
        x = start_point[0] + i * step_x
        if end_point[0] - start_point[0] == 0:
            # print('here')
            return None
        y = start_point[1] + (x - start_point[0]) * (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        points.append([x, y])

    points.append([end_point[0], end_point[1]])
    return points 



class SimpleKPLoader(object):
    def __init__(self, root, image_size, image_set='test', data_set='tusimple', norm=False):

        self.image_set = image_set
        self.data_set = data_set
        self.root = root
        self.norm = norm
        self.image_height = image_size[0]
        self.image_width = image_size[-1]

        if self.image_set == 'test' and data_set == 'llamas':
            raise ValueError

        if data_set == 'tusimple':
            self.image_dir = root
        elif data_set == 'culane':
            self.image_dir = root
            self.annotations_suffix = '.lines.txt'
        elif data_set == 'curvelanes':
            self.image_dir = root
            self.annotations_suffix = '.lines.txt'
        elif data_set == 'llamas':
            self.image_dir = os.path.join(root, 'color_images')
            self.annotations_suffix = '.lines.txt'
        elif data_set == 'liquid':
            print('generate Liquid dataset bezier control keypoint.')
        else:
            raise ValueError

        self.splits_dir = os.path.join(root, 'lists')

    def load_txt_path(self, dataset):
        split_f = os.path.join(self.splits_dir, self.image_set + '.txt')
        with open(split_f, "r") as f:
            contents = ['d' + x.strip() for x in f.readlines()]
        if self.image_set in ['test', 'val']:
            path_lists = [os.path.join(self.image_dir, x + self.annotations_suffix) for x in contents]
        elif self.image_set == 'train':
            if dataset == 'curvelanes':
                path_lists = [os.path.join(self.image_dir, x + self.annotations_suffix) for x in contents]
            else:
                path_lists = [os.path.join(self.image_dir, x[:x.find(' ')] +
                                           self.annotations_suffix) for x in contents]
        else:
            raise ValueError

        return path_lists

    def load_json(self):
        if self.image_set == 'test':
            json_name = [os.path.join(self.image_dir, 'test_label.json')]
        elif self.image_set == 'val':
            json_name = [os.path.join(self.image_dir, 'label_data_0531.json')]
        elif self.image_set == 'train':
            json_name = [os.path.join(self.image_dir, 'label_data_0313.json'),
                            os.path.join(self.image_dir, 'label_data_0601.json')]
        else:
            raise ValueError

        return json_name

    def get_points_in_txtfile(self, file_path):
        coords = []
        with open(file_path, 'r') as f:
            for lane in f.readlines():
                lane = lane.split(' ')
                coord = []
                for i in range(0, len(lane) - 1, 2):
                    if float(lane[i]) >= 0:
                        coord.append([float(lane[i]), float(lane[i + 1])])
                coord = np.array(coord)
                if self.norm and len(coord) != 0:
                    coord[:, 0] = coord[:, 0] / self.image_width
                    coord[:, -1] = coord[:, -1] / self.image_height
                coords.append(coord)

        return coords

    def get_points_in_json(self, itm):
        lanes = itm['lanes']
        coords_list = []
        h_sample = itm['h_samples']
        for lane in lanes:
            coord = []
            for x, y in zip(lane, h_sample):
                if x >= 0:
                    coord.append([float(x), float(y)])
            coord = np.array(coord)
            if self.norm and len(coord) != 0:
                coord[:, 0] = coord[:, 0] / self.image_width
                coord[:, -1] = coord[:, -1] / self.image_height
            coords_list.append(coord)

        return coords_list

    def load_annotations(self):
        print('Loading dataset...')
        coords = OrderedDict()
        if self.data_set in ['culane', 'llamas', 'curvelanes']:
            file_lists = self.load_txt_path(dataset=self.data_set)
            for f in tqdm(file_lists):
                coords[f[len(self.root) + 1:]] = self.get_points_in_txtfile(f)
        elif self.data_set == 'tusimple':
            jsonfiles = self.load_json()

            results = []
            for jsonfile in jsonfiles:
                with open(jsonfile, 'r') as f:
                    results += [json.loads(x.strip()) for x in f.readlines()]
            for lane_json in tqdm(results):
                coords[lane_json['raw_file']] = self.get_points_in_json(lane_json)
        elif self.data_set == 'liquid':
            coords = self.load_annotations_liquid()
        else:
            raise ValueError
        print('Finished')

        return coords
    
    def load_annotations_liquid(self):
        if self.image_set == 'train':
            visib_threshold = 0.2
        else:
            visib_threshold = 0.1
        root_dir = os.path.join(self.root, self.image_set)
        coords = OrderedDict()
        for scene in os.listdir(root_dir):
            current_dir = os.path.join(root_dir, scene)
            scene_gt_fn = os.path.join(current_dir, 'scene_gt_liquid.json')
            scene_gt_info_fn = os.path.join(current_dir, "scene_gt_info.json")
            if not os.path.exists(scene_gt_fn) or not os.path.exists(scene_gt_info_fn):
                continue

            scene_gts = load_scene_gt(scene_gt_fn)
            scene_gt_infos = load_scene_gt(scene_gt_info_fn)

            for img_id in sorted(scene_gts.keys()):
                img_id = int(img_id)
                if len(scene_gts[img_id]) != len(scene_gt_infos[img_id]):
                    continue

                for rank, gt in enumerate(scene_gts[img_id]):
                    obj_id = int(gt['obj_id'])
                    gt_info = scene_gt_infos[img_id][rank]

                    visib_fract = scene_gt_infos[img_id][rank]['visib_fract']
                    px_count_visib = scene_gt_infos[img_id][rank]['px_count_visib']
                    if not(visib_fract > visib_threshold and px_count_visib >= 1600):
                        continue

                    mask_fn = os.path.join(current_dir, 'mask', f'{img_id:06d}_{rank:06d}.png')
                    liquid_label = gt['liquid_label']
                    bbox = gt_info['bbox_visib']

                    points = process_lanes(obj_id, liquid_label)
                    if points.any() == np.array(None):
                        continue
                    #################
                    # rgb_fn = mask_fn.replace('mask', 'rgb').split('_')[0] + '.png'
                    # rgb = cv2.imread(rgb_fn)

                    # rgb = rgb[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
                    # scale_factor, rgb = resize_img(rgb, 512)

                    points = del_double(points.tolist())
                    points = interp(points, n=20)

                    scale_factor = 512 / max(bbox[2], bbox[3])
                    points[:, 0] -= bbox[0]
                    points[:, 1] -= bbox[1]
                    points = points * scale_factor
                    points = np.array(sorted(points, key=lambda point: point[0]))

                    # for p in points:
                    #     rgb = cv2.circle(rgb, (int(p[0]), int(p[1])), 1, (255, 0, 0), -1)
                    # cv2.imwrite('debug/points.png', rgb)

                    # fcns = BezierCurve(order=3)
                    # fcns.get_control_points(points[:, 0], points[:, 1])
                    # bezier = fcns.save_control_points()
                    # for p in bezier:
                    #     rgb = cv2.circle(rgb, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
                    # cv2.imwrite('debug/bezier.png', rgb)

                    # samples = fcns.quick_sample_point()
                    # for p in samples:
                    #     rgb = cv2.circle(rgb, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
                    # cv2.imwrite('debug/bezier_points.png', rgb)

                    coords[mask_fn] = [points]
        return coords
    

    def concat_jsons(self, filenames):
        # Concat tusimple lists in jsons (actually only each line is json)
        results = []
        for filename in filenames:
            with open(filename, 'r') as f:
                results += [json.loads(x.strip()) for x in f.readlines()]

        return results
