import time
import os
import argparse

import torch
import torchvision
import tensorrt as trt
from torch2trt import torch2trt
import matplotlib.pyplot as plt

from data import datasets
from model import loader
from metrics import AverageMeter, Result
from data import transforms

import tent
from conf import cfg, load_cfg_fom_args
import torch.optim as optim

max_depths = {
    'kitti': 80.0,
    'nyu': 10.0,
    'nyu_reduced': 10.0,
}
nyu_res = {
    'full': (480, 640),
    'half': (240, 320),
    'mini': (224, 224)}
kitti_res = {
    'full': (384, 1280),
    'half': (192, 640)}
resolutions = {
    'nyu': nyu_res,
    'nyu_reduced': nyu_res,
    'kitti': kitti_res}
crops = {
    'kitti': [128, 381, 45, 1196],
    'nyu': [20, 460, 24, 616],
    'nyu_reduced': [20, 460, 24, 616]}


def get_args():
    parser = argparse.ArgumentParser(description='Nano Inference for Monocular Depth Estimation')

    # Mode
    parser.set_defaults(evaluate=False)
    parser.add_argument('--eval',
                        dest='evaluate',
                        action='store_true')

    # Data
    parser.add_argument('--test_path',
                        type=str,
                        help='path to test data')
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset for training',
                        choices=['kitti', 'nyu', 'nyu_reduced'],
                        default='kitti')
    parser.add_argument('--resolution',
                        type=str,
                        help='Resolution of the images for training',
                        choices=['full', 'half'],
                        default='full')

    # Model
    parser.add_argument('--model',
                        type=str,
                        help='name of the model to be trained',
                        default='UpDepth')
    parser.add_argument('--weights_path',
                        type=str,
                        help='path to model weights')
    parser.add_argument('--save_results',
                        type=str,
                        help='path to save results to',
                        default='./results')

    # System
    parser.add_argument('--num_workers',
                        type=int,
                        help='number of dataloader workers',
                        default=1)

    return parser.parse_args()


class Inference_Engine():
    def __init__(self, args):
        self.maxDepth = max_depths[args.dataset]
        self.res_dict = resolutions[args.dataset]
        self.resolution = self.res_dict[args.resolution]
        self.resolution_keyword = args.resolution
        print('Resolution for Eval: {}'.format(self.resolution))
        print('Maximum Depth of Dataset: {}'.format(self.maxDepth))
        self.crop = crops[args.dataset]

        self.result_dir = args.save_results
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.results_filename = '{}_{}_{}'.format(args.dataset,
                                                  args.resolution,
                                                  args.model)

        self.device = torch.device('cuda:0')

        # self.model = loader.load_model(args.model, args.weights_path)
        # self.model = self.model.eval().cuda()

        self.tented_model = setup_tent(loader.load_model(args.model, args.weights_path)).cuda()

        self.tented_model = self.tented_model.eval()

        if args.evaluate:
            self.test_loader = datasets.get_dataloader(args.dataset,
                                                       path=args.test_path,
                                                       split='test',
                                                       batch_size=1,
                                                       resolution=args.resolution,
                                                       uncompressed=True,
                                                       workers=args.num_workers)

        if args.resolution == 'half':
            self.upscale_depth = torchvision.transforms.Resize(self.res_dict['full'])  # To Full res
            self.downscale_image = torchvision.transforms.Resize(self.resolution)  # To Half res

        self.to_tensor = transforms.ToTensor(test=True, maxDepth=self.maxDepth)

        self.visualize_images = [0]

        # self.trt_model, _ = self.convert_PyTorch_to_TensorRT()

        self.run_evaluation()

    def run_evaluation(self):

        average = self.tensorRT_evaluate()


    def tensorRT_evaluate(self):
        torch.cuda.empty_cache()
        # self.model = None
        average_meter = AverageMeter()

        dataset = self.test_loader.dataset
        for i, data in enumerate(dataset):
            t0 = time.time()
            image, gt = data
            packed_data = {'image': image, 'depth': gt}
            data = self.to_tensor(packed_data)
            image, gt = self.unpack_and_move(data)
            image = image.unsqueeze(0)
            gt = gt.unsqueeze(0)

            image_flip = torch.flip(image, [3])
            gt_flip = torch.flip(gt, [3])
            if self.resolution_keyword == 'half':
                image = self.downscale_image(image)
                image_flip = self.downscale_image(image_flip)

            torch.cuda.synchronize()
            data_time = time.time() - t0

            t0 = time.time()
            inv_prediction = self.tented_model(image)
            prediction = self.inverse_depth_norm(inv_prediction)
            torch.cuda.synchronize()
            gpu_time0 = time.time() - t0

            t1 = time.time()
            inv_prediction_flip = self.tented_model(image_flip)
            prediction_flip = self.inverse_depth_norm(inv_prediction_flip)
            torch.cuda.synchronize()
            gpu_time1 = time.time() - t1

            if self.resolution_keyword == 'half':
                prediction = self.upscale_depth(prediction)
                prediction_flip = self.upscale_depth(prediction_flip)

            if i in self.visualize_images:
                self.save_image_results(image, gt, prediction, i)

            gt = gt[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
            gt_flip = gt_flip[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
            prediction = prediction[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
            prediction_flip = prediction_flip[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]

            result = Result()
            result.evaluate(prediction.data, gt.data)
            average_meter.update(result, gpu_time0, data_time, image.size(0))

            result_flip = Result()
            result_flip.evaluate(prediction_flip.data, gt_flip.data)
            average_meter.update(result_flip, gpu_time1, data_time, image.size(0))

        # Report
        avg = average_meter.average()
        current_time = time.strftime('%H:%M', time.localtime())
        print('\n*\n'
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'Delta1={average.delta1:.3f}\n'
              'Delta2={average.delta2:.3f}\n'
              'Delta3={average.delta3:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
              't_GPU={time:.3f}\n'.format(
            average=avg, time=avg.gpu_time))
        return avg

    def inverse_depth_norm(self, depth):
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        return depth

    def depth_norm(self, depth):
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        return depth

    def unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            return image, gt
        if isinstance(data, dict):
            keys = data.keys()
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            return image, gt
        print('Type not supported')

    def save_image_results(self, image, gt, prediction, image_id):
        img = image[0].permute(1, 2, 0).cpu()
        gt = gt[0, 0].permute(0, 1).cpu()
        prediction = prediction[0, 0].permute(0, 1).detach().cpu()
        error_map = gt - prediction
        vmax_error = self.maxDepth / 10.0
        vmin_error = 0.0
        cmap = 'viridis'

        vmax = torch.max(gt[gt != 0.0])
        vmin = torch.min(gt[gt != 0.0])

        save_to_dir = os.path.join(self.result_dir, 'inference_image_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'inference_errors_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        errors = ax.imshow(error_map, vmin=vmin_error, vmax=vmax_error, cmap='Reds')
        fig.colorbar(errors, ax=ax, shrink=0.8)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'inference_gt_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(gt, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'inference_depth_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(prediction, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.savefig(save_to_dir)
        plt.clf()


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    return tent_model      # tent_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


if __name__ == '__main__':
    args = get_args()
    print(args)

    engine = Inference_Engine(args)
