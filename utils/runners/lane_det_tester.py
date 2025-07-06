import os
import cv2
import numpy as np
import torch

try:
    import ujson as json
except ImportError:
    import json
from tqdm import tqdm
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from ..torch_amp_dummy import autocast

from .base import BaseTester
from ..seg_utils import ConfusionMatrix
from ..lane_det_utils import lane_as_segmentation_inference


class LaneDetTester(BaseTester):
    image_sets = ['valfast', 'test', 'val']

    def __init__(self, cfg):
        super().__init__(cfg)
        self.fast_eval = True if self._cfg['state'] == 1 else False

    def run(self):
        if self.fast_eval:
            _, x = self.fast_evaluate(self.model, self.device, self.dataloader,
                                      self._cfg['mixed_precision'], self._cfg['input_size'], self._cfg['num_classes'])
            self.write_mp_log('log.txt', self._cfg['exp_name'] + ' validation: ' + str(x) + '\n')
        else:
            self.test_one_set(self.model, self.device, self.dataloader, self._cfg['mixed_precision'],
                              [self._cfg['input_size'], self._cfg['original_size']],
                              self._cfg['gap'], self._cfg['ppl'], self._cfg['thresh'],
                              self._cfg['dataset_name'], self._cfg['seg'], self._cfg['max_lane'], self._cfg['exp_name'])

    @staticmethod
    @torch.no_grad()
    def test_one_set(net, device, loader, mixed_precision, input_sizes, gap, ppl, thresh, dataset,
                     seg, max_lane=0, exp_name=None):
        # Adapted from harryhan618/SCNN_Pytorch
        # Predict on 1 data_loader and save predictions for the official script
        # sizes: [input size, test original size, ...]
        # max_lane = 0 -> unlimited number of lanes

        save_color = False
        save_features = False
        save_dir = '/home/wxy/exps/Bezier_liquid/bezier_color_ffb/inter_output'
        # save_dir = '/home/wxy/exps/debug_colornet'

        all_lanes = []
        net.eval()
        preds = {}
        for images, filenames in tqdm(loader):
            images = images.to(device)
            with autocast(mixed_precision):
                if seg:
                    batch_coordinates = lane_as_segmentation_inference(net, images,
                                                                       input_sizes, gap, ppl, thresh, dataset, max_lane)
                else:
                    batch_coordinates, colored_images, color_residuals, features = net.inference(images, input_sizes, gap, ppl, dataset, max_lane)
            if save_color:
                assert images.shape[0] == 1, "Batch size have to be 1 ."
                filename = filenames[0]
                # if filename.split('/')[-3] == '000000':
                scene = filename.split('/')[-3]
                f_name = filename.split('/')[-1].replace('.png', '.npy')

                if not os.path.exists(os.path.join(save_dir, 'colored_image', scene)):
                    os.makedirs(os.path.join(save_dir, 'colored_image', scene), exist_ok=True)
                colored_image_fn = os.path.join(save_dir, 'colored_image', scene, f_name)
                colored_image = colored_images[0].permute(1, 2, 0).cpu().numpy()
                np.save(colored_image_fn, colored_image)
                
                if not os.path.exists(os.path.join(save_dir, 'color_residual', scene)):
                    os.makedirs(os.path.join(save_dir, 'color_residual', scene), exist_ok=True)
                color_residual_fn = os.path.join(save_dir, 'color_residual', scene, f_name)
                color_residual = color_residuals[0].permute(1, 2, 0).cpu().numpy()
                np.save(color_residual_fn, color_residual)
            
            if save_features:
                assert features.shape[0] == 1, "Batch size have to be 1"
                filename = filenames[0]
                # if filename.split('/')[-3] == '000000':
                scene = filename.split('/')[-3]
                f_name = filename.split('/')[-1].replace('.png', '.pth')

                if not os.path.exists(os.path.join(save_dir, 'features', scene)):
                    os.makedirs(os.path.join(save_dir, 'features', scene), exist_ok=True)
                features_fn = os.path.join(save_dir, 'features', scene, f_name)
                torch.save(features[0], features_fn)

            # Parse coordinates
            for j in range(len(batch_coordinates)):
                lane_coordinates = batch_coordinates[j]
                if dataset == 'culane':
                    # Save each lane to disk
                    dir_name = filenames[j][:filenames[j].rfind('/')]
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    with open(filenames[j], "w") as f:
                        for lane in lane_coordinates:
                            if lane:  # No printing for []
                                for (x, y) in lane:
                                    print("{} {}".format(x, y), end=" ", file=f)
                                print(file=f)
                elif dataset == 'liquid':
                    fn = filenames[j]
                    preds.update({fn: lane_coordinates})

                elif dataset == 'tusimple':
                    # Save lanes to a single file
                    formatted = {
                        "h_samples": [160 + y * 10 for y in range(ppl)],
                        "lanes": [[c[0] for c in lane] for lane in lane_coordinates],
                        "run_time": 0,
                        "raw_file": filenames[j]
                    }
                    all_lanes.append(json.dumps(formatted))
                elif dataset == 'llamas':
                    # save each lane in images in xxx.lines.txt
                    dir_name = filenames[j][:filenames[j].rfind('/')]
                    file_path = filenames[j].replace("_color_rect", "")
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    with open(file_path, "w") as f:
                        for lane in lane_coordinates:
                            if lane:  # No printing for []
                                for (x, y) in lane:
                                    print("{} {}".format(x, y), end=" ", file=f)
                                print(file=f)
                else:
                    raise ValueError

        if dataset == 'tusimple':
            with open('./output/' + exp_name + '.json', 'w') as f:
                for lane in all_lanes:
                    print(lane, end="\n", file=f)
        
        elif dataset == 'liquid':
            save_fn = os.path.join('output', exp_name+'_preds.json')
            with open(save_fn, 'w') as f:
                json.dump(preds, f, indent=4)
                print(f'All predictions are saved in {save_fn}')

    @staticmethod
    @torch.no_grad()
    def fast_evaluate(net, device, loader, mixed_precision, output_size, num_classes):
        # Fast evaluation (e.g. on the validation set) by pixel-wise mean IoU
        net.eval()
        conf_mat = ConfusionMatrix(num_classes)
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            with autocast(mixed_precision):
                output = net(image)['out']
                output = torch.nn.functional.interpolate(output, size=output_size,
                                                         mode='bilinear', align_corners=True)
                conf_mat.update(target.flatten(), output.argmax(1).flatten())
        conf_mat.reduce_from_all_processes()

        acc_global, acc, iu = conf_mat.compute()
        print((
            'global correct: {:.2f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}'
        ).format(
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100))

        return acc_global.item() * 100, iu.mean().item() * 100
