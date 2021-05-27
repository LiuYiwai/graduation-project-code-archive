"""
随机组合超像素输入关系网络判断未标记超像素的类别
"""
import os.path as osp
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.segmentation import slic
import torchvision

from utils import empty_tensor
from utils import is_empty_tensor
from utils.data import SegmentationDataset
from utils.data import Digest2019PointDataset
from .base import BaseConfig, BaseTrainer


def _preprocess_superpixels(segments, mask=None, epsilon=1e-7):
    """Segment superpixels of a given image and return segment maps and their labels.

    Args:
        segments: slic segments tensor with shape (H, W)
        mask (optional): annotation mask tensor with shape (C, H, W). Each pixel is a one-hot
            encoded label vector. If this vector is all zeros, then its class is unknown.
    Returns:
        sp_maps: superpixel maps with shape (N, H, W)
        sp_labels: superpixel labels with shape (N_l, C), where N_l is the number of labeled samples.
    """

    # ordering of superpixels
    sp_idx_list = segments.unique()

    if mask is not None and not is_empty_tensor(mask):
        def compute_superpixel_label(sp_idx):
            sp_mask = (mask * (segments == sp_idx).long()).float()
            return sp_mask.sum(dim=(1, 2)) / (sp_mask.sum() + epsilon)

        # compute labels for each superpixel
        sp_labels = torch.cat([
            compute_superpixel_label(sp_idx).unsqueeze(0)
            for sp_idx in range(segments.max() + 1)
        ])

        # move labeled superpixels to the front of `sp_idx_list`
        labeled_sps = (sp_labels.sum(dim=-1) > 0).nonzero().flatten()
        unlabeled_sps = (sp_labels.sum(dim=-1) == 0).nonzero().flatten()
        sp_idx_list = torch.cat([labeled_sps, unlabeled_sps])

        # quantize superpixel labels (e.g., from (0.7, 0.3) to (1.0, 0.0))
        sp_labels = sp_labels[labeled_sps]
        sp_labels = (sp_labels == sp_labels.max(
            dim=-1, keepdim=True)[0]).float()
    else:  # no supervision provided
        sp_labels = empty_tensor().to(segments.device)

    # stacking normalized superpixel segment maps
    sp_maps = segments == sp_idx_list[:, None, None]
    sp_maps = sp_maps.squeeze().float()

    # make sure each superpixel map sums to one
    sp_maps = sp_maps / sp_maps.sum(dim=(1, 2), keepdim=True)

    return sp_maps, sp_labels


def _cross_entropy(y_hat, y_true, class_weights=None, pseudo_weight=1, epsilon=1e-7):
    """Semi-supervised cross entropy loss function.

    Args:
        y_hat: prediction tensor with size (N, C), where C is the number of classes
        y_true: label tensor with size (N, C). A sample won't be counted into loss
            if its label is all zeros.
        class_weights: class weights tensor with size (C,)
        epsilon: numerical stability term

    Returns:
        cross_entropy: cross entropy loss computed only on samples with labels
    """
    # clamp all elements to prevent numerical overflow/underflow
    # [1, H, W]
    y_hat = torch.clamp(y_hat, min=epsilon, max=(1 - epsilon))

    y_hat = y_hat.view(-1)
    y_true = y_true.view(-1)

    def get_one_hot(src):
        target = 1 - src
        return torch.cat((target.unsqueeze(1), src.unsqueeze(1)), 1)

    def get_loss(y_hat, y_true, class_weights):
        y_hat = get_one_hot(y_hat)
        y_true = get_one_hot(y_true)
        ce = -y_true * torch.log(y_hat)
        if class_weights is not None:
            ce = ce * class_weights.unsqueeze(0).float()
        loss = torch.sum(ce) / y_true.size(0)
        return loss

    compute_loss_samples = y_true >= 0
    y_hat = y_hat[compute_loss_samples]
    y_true = y_true[compute_loss_samples]

    gt_loss_mask = torch.where((y_true == 0) == True, y_true == 0, y_true == 1)
    pseudo_loss_mask = ~gt_loss_mask
    y_true[y_true >= 0.5] = 1
    y_true[y_true < 0.5] = 0

    gt_y_hat = y_hat[gt_loss_mask]
    gt_y_true = y_true[gt_loss_mask]
    loss = get_loss(gt_y_hat, gt_y_true, class_weights)

    if torch.sum(pseudo_loss_mask) > 0:
        pseudo_y_hat = y_hat[pseudo_loss_mask]
        pseudo_y_true = y_true[pseudo_loss_mask]
        loss += pseudo_weight * \
            get_loss(pseudo_y_hat, pseudo_y_true, class_weights)

    return loss


def _label_propagate(features, y_l, threshold=0.95):
    """Perform random walk based label propagation with similarity graph.

    Arguments:
        features: features of size (N, D), where N is the number of superpixels
            and D is the dimension of input features
        y_l: label tensor of size (N, C), where C is the number of classes
        threshold: similarity threshold for label propagation

    Returns:
        pseudo_labels: propagated label tensor of size (N, C)
    """

    # disable gradient computation
    features = features.detach()
    y_l = y_l.detach()

    # number of labeled and unlabeled samples
    n_l = y_l.size(0)
    n_u = features.size(0) - n_l

    # feature affinity matrix
    W = torch.exp(-torch.einsum('ijk,ijk->ij',
                                features - features.unsqueeze(1),
                                features - features.unsqueeze(1)))

    # sub-matrix of W containing similarities between labeled and unlabeled samples
    W_ul = W[n_l:, :n_l]

    # max_similarities is the maximum similarity for each unlabeled sample
    # src_indexes is the respective labeled sample index
    max_similarities, src_indexes = W_ul.max(dim=1)

    # initialize y_u with zeros
    y_u = torch.zeros(n_u, y_l.size(1)).to(y_l.device)

    # only propagate labels if maximum similarity is above the threshold
    propagated_samples = max_similarities > threshold
    y_u[propagated_samples] = y_l[src_indexes[propagated_samples]]

    return y_u


class WESUPConfig(BaseConfig):
    """Configuration for WESUP model."""

    # Rescale factor to subsample input images.
    rescale_factor = 0.5

    # multi-scale range for training
    multiscale_range = (0.4, 0.6)

    # Number of target classes.
    n_classes = 2

    # Class weights for cross-entropy loss function.
    class_weights = (1, 1)
    # class_weights = (3, 1)

    mask_loss_weight = 1
    relation_loss_weight = 1
    # relation_loss_weight = 2

    # Superpixel parameters.
    sp_area = 50
    # sp_area = 200
    sp_compactness = 40

    # whether to enable label propagation
    enable_propagation = True

    # Weight for label-propagated samples when computing loss function
    propagate_threshold = 0.8

    # Weight for label-propagated samples when computing loss function
    # propagate_weight = 0.5
    propagate_weight = 0.6

    # Optimization parameters.
    momentum = 0.9
    weight_decay = 0.001

    # Whether to freeze backbone.
    freeze_backbone = False

    # Training configurations.  batch_size = 1
    epochs = 300


class WESUP(nn.Module):
    """Weakly supervised histopathology image segmentation with sparse point annotations."""

    def __init__(self, n_classes=2, D=32, **kwargs):
        """Initialize a WESUP model.

        Kwargs:
            n_classes: number of target classes (default to 2)
            D: output dimension of superpixel features

        Returns:
            model: a new WESUP model
        """

        super().__init__()

        self.kwargs = kwargs
        self.fm_channels_sum = 0

        self.backbone = models.vgg16(pretrained=True).features
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)
                setattr(self, f'side_conv{self.fm_channels_sum}',
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1))

        self.backbone_relation = models.vgg16(pretrained=True).features
        for layer in self.backbone_relation:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)
                setattr(self, f'side_conv{self.fm_channels_sum}',
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1))
                self.fm_channels_sum += layer.out_channels // 2

        D = 32
        inter_feature = 1024
        self.feature_classifier = nn.Sequential(
            nn.Linear(self.fm_channels_sum, inter_feature),
            nn.ReLU(inplace=True),
            nn.Linear(inter_feature, inter_feature),
            nn.ReLU(inplace=True),
            nn.Linear(inter_feature, D),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(D, self.kwargs.get('n_classes', 2)),
            nn.Softmax(dim=1)
        )

        D = 512
        self.feature_relation = nn.Sequential(
            nn.Linear(self.fm_channels_sum, inter_feature),
            nn.ReLU(inplace=True),
            nn.Linear(inter_feature, inter_feature),
            nn.ReLU(inplace=True),
            nn.Linear(inter_feature, D),
            nn.ReLU(inplace=True),
        )
        self.relation_classifier = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.ReLU(inplace=True),
            nn.Linear(D, self.kwargs.get('n_classes', 2)),
            nn.Softmax(dim=1)
        )

        # store conv feature maps
        self.feature_maps = None

        # spatial size of first feature map
        self.fm_size = None

        # pixel predictions (tracked to compute loss)
        self.p_pred = None

        # label propagation input features
        self.p_features = None

        # label propagation input features
        self.sp_features = None

    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            self.fm_size = (input_[0].size(2), input_[0].size(3))
            side_conv_name = 'side_conv0'
        else:
            side_conv_name = f'side_conv{self.feature_maps.size(0)}'

        output = getattr(self, side_conv_name)(output.clone())
        output = F.interpolate(output, self.fm_size,
                               mode='bilinear', align_corners=True)

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def _pred_relation_forward(self, sp_labels, sp_features,
                               pseudo_train_labels, pseudo_train_features):

        def get_relation_forward(sp_labels, sp_features):
            base_class = 1
            base_class_index = torch.where(sp_labels[:, 1] == base_class)[0]

            if len(base_class_index) == 0:
                return None, None

            sp_num = sp_labels.size(0)
            # train_relation_weight = min(16., sp_num)
            train_relation_weight = 2.

            # train sample num
            train_num = int(sp_num * train_relation_weight)
            first_index = np.random.choice(
                len(base_class_index), size=train_num)
            second_pick_index = np.random.choice(
                sp_labels.size(0), size=train_num)
            first_pick_index = base_class_index[first_index]

            first_features = sp_features[first_pick_index]
            second_features = sp_features[second_pick_index]

            inp = torch.cat((first_features, second_features), 1)
            pred = self.relation_classifier(inp)[:, 1]
            target = sp_labels[second_pick_index][:, 1]

            return pred, target

        relation_pred, relation_target = get_relation_forward(sp_labels, sp_features)

        if pseudo_train_labels is not None:
            pseudo_train_labels = torch.cat((pseudo_train_labels, sp_labels), 0)
            pseudo_train_features = torch.cat((pseudo_train_features, sp_features), 0)
            pseudo_relation_pred, pseudo_relation_target = get_relation_forward(pseudo_train_labels,
                                                                                pseudo_train_features)
        else:
            pseudo_relation_pred = None
            pseudo_relation_target = None

        return relation_pred, relation_target, \
            pseudo_relation_pred, pseudo_relation_target

    def _get_pseudo_lables(self, sp_maps, sp_labels, sp_features, p_features):
        device = sp_labels.device
        height, width = sp_maps.size()
        n_l = sp_labels.size(0)

        top1 = 0.9
        top2 = 0.8
        bottom1 = 0.1
        bottom2 = 0.2
        base_class = 1

        positive_pseudo_tag = 0.8
        negative_pseudo_tag = 0.2

        pseudo_labels = torch.zeros(height * width).to(device).detach()
        sp_maps = sp_maps.view(height * width)

        # GT
        for sp_idx in range(n_l):
            pseudo_labels[sp_maps == sp_idx] = sp_labels[sp_idx][1]

        base_class_index = torch.where(sp_labels[:, 1] == base_class)[0]
        base_class_featrues = sp_features[base_class_index]
        base_class_num = len(base_class_index)

        unlabeled_pixel_mask = (sp_maps >= n_l).view(p_features.size(0))
        unlabeled_pixel_mask_index = torch.nonzero(unlabeled_pixel_mask)[:, 0]
        unlabeled_pixel_features = p_features[unlabeled_pixel_mask]
        unlabeled_pixel_num = unlabeled_pixel_features.size(0)

        first_index = np.random.choice(base_class_num, size=unlabeled_pixel_num)
        first_features = base_class_featrues[first_index]
        inp = torch.cat((first_features, unlabeled_pixel_features), 1)
        pred = self.relation_classifier(inp)[:, 1]

        # add pseudo
        positive_mask = pred > top1
        pseudo_positive_features = p_features[unlabeled_pixel_mask_index[positive_mask]]
        negative_mask = pred < bottom1
        pseudo_negative_features = p_features[unlabeled_pixel_mask_index[negative_mask]]

        # add tag
        positive_add_label = pred > top2
        positive_add_label_index = unlabeled_pixel_mask_index[positive_add_label]
        pseudo_labels[positive_add_label_index] = positive_pseudo_tag
        negative_add_label = pred < bottom2
        negative_add_label_index = unlabeled_pixel_mask_index[negative_add_label]
        pseudo_labels[negative_add_label_index] = negative_pseudo_tag

        # recheck
        recheck_mask = ~(positive_add_label + negative_add_label)
        recheck_index = unlabeled_pixel_mask_index[recheck_mask]
        recheck_num = len(recheck_index)
        recheck_features = p_features[recheck_index]

        if base_class == 1:
            pseudo_base_class_featrues = torch.cat((base_class_featrues,
                                                    pseudo_positive_features), 0)
        else:
            pseudo_base_class_featrues = torch.cat((base_class_featrues,
                                                    pseudo_negative_features), 0)

        if recheck_num > 0:
            first_index = np.random.choice(pseudo_base_class_featrues.size(0), size=recheck_num)
            first_features = pseudo_base_class_featrues[first_index]
            inp = torch.cat((first_features, recheck_features), 1)
            pred = self.relation_classifier(inp)[:, 1]
    
            # add pseudo
            positive_mask = pred > top1
            pseudo_positive_features = torch.cat((pseudo_positive_features,
                                                  p_features[recheck_index[positive_mask]]), 0)
            negative_mask = pred < bottom1
            pseudo_negative_features = torch.cat((pseudo_negative_features,
                                                  p_features[recheck_index[negative_mask]]), 0)
    
            # add tag
            positive_add_label = pred > top2
            pseudo_labels[recheck_index[positive_add_label]] = positive_pseudo_tag
            negative_add_label = pred < bottom2
            pseudo_labels[recheck_index[negative_add_label]] = negative_pseudo_tag
    
            comfuse_mask = ~(positive_add_label + negative_add_label)
            comfuse_positive_mask = (pred > 0.5) & comfuse_mask
            comfuse_negative_mask = (pred < 0.5) & comfuse_mask
    
            comfuse_positive_index = recheck_index[comfuse_positive_mask]
            comfuse_negative_index = recheck_index[comfuse_negative_mask]
            pseudo_labels[comfuse_positive_index] = -1
            pseudo_labels[comfuse_negative_index] = -2
    
        pseudo_labels = pseudo_labels.view(height, width)
        pseudo_positive_num = pseudo_positive_features.shape[0]
        pseudo_negative_num = pseudo_negative_features.shape[0]

        if pseudo_positive_num + pseudo_negative_num == 0:
            pseudo_train_labels = None
            pseudo_train_features = None
        else:
            pseudo_positive_labels = torch.tensor((0, 1)).repeat(pseudo_positive_num, 1).to(device)
            pseudo_negative_labels = torch.tensor((1, 0)).repeat(pseudo_negative_num, 1).to(device)

            pseudo_train_labels = torch.cat((pseudo_positive_labels,
                                             pseudo_negative_labels), 0)
            pseudo_train_features = torch.cat((pseudo_positive_features,
                                               pseudo_negative_features), 0)

        # for_img = pseudo_labels
        # for_img[for_img == -1] = 0.75
        # for_img[for_img == -2] = 0.25
        # for_img[for_img == 0.8] = 1
        # for_img[for_img == 0.2] = 0
        # img = (for_img * 255).int()
        # img = torchvision.transforms.ToPILImage()(img).convert('L')
        # img.show()

        return pseudo_labels, pseudo_train_labels, pseudo_train_features

    def forward(self, x):
        """Running a forward pass.

        Args:
            x: a tuple containing input tensor of size (1, C, H, W) and
                stacked superpixel maps with size (N, H, W)

        Returns:
            pred: prediction with size (1, H, W)
        """

        x, sp_maps, sp_labels = x
        n_superpixels, height, width = sp_maps.size()

        # x_img = x
        # img = x_img[0].float()
        # img = torchvision.transforms.ToPILImage()(img * 255).convert('L')
        # img.show()

        raw_x = x

        self.feature_maps = None
        _ = self.backbone(raw_x)
        x_class = self.feature_maps
        is_train = x_class.requires_grad
        x = x_class.view(x_class.size(0), -1)
        p_x = x.t()
        class_p_x = self.feature_classifier(p_x)
        self.p_pred = self.classifier(class_p_x)
        pred = self.p_pred[..., 1].view(height, width).unsqueeze(0)

        # img = (pred[0] > 0.5).int()
        # img = torchvision.transforms.ToPILImage()(img * 255).convert('L')
        # img.show()

        self.feature_maps = None
        _ = self.backbone_relation(raw_x)
        x_relation = self.feature_maps

        # calculate features for each superpixel
        x = x_relation.view(x_relation.size(0), -1)
        sp_maps = sp_maps.view(sp_maps.size(0), -1)
        sp_x = torch.mm(sp_maps, x.t())
        sp_maps = sp_maps.view(n_superpixels, height, width).argmax(dim=0)
        # and pixel
        p_x = x.t()

        if is_train:
            relation_p_x = self.feature_relation(p_x)
            relation_sp_x = self.feature_relation(sp_x)

            pseudo_labels, pseudo_train_labels, pseudo_train_features = self._get_pseudo_lables(
                sp_maps, sp_labels, relation_sp_x, relation_p_x)

            sp_relation_pred, sp_relation_target,\
                pseudo_relation_pred, pseudo_relation_target = self._pred_relation_forward(
                    sp_labels, relation_sp_x,
                    pseudo_train_labels, pseudo_train_features)
        else:
            pseudo_labels = None
            sp_relation_pred, sp_relation_target = None, None
            pseudo_relation_pred, pseudo_relation_target = None, None

        return pred, pseudo_labels, sp_relation_pred, sp_relation_target, \
            pseudo_relation_pred, pseudo_relation_target


class WESUPPixelInference(WESUP):
    """Weakly supervised histopathology image segmentation with sparse point annotations."""

    def __init__(self, n_classes=2, D=32, **kwargs):
        # def __init__(self, n_classes=2, D=64, **kwargs):
        """Initialize a WESUP model.

        Kwargs:
            n_classes: number of target classes (default to 2)
            D: output dimension of superpixel features

        Returns:
            model: a new WESUP model
        """

        super().__init__()

        self.kwargs = kwargs
        self.backbone = models.vgg16(pretrained=True).features

        # sum of channels of all feature maps
        self.fm_channels_sum = 0

        # side convolution layers after each conv feature map
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)
                setattr(self, f'side_conv{self.fm_channels_sum}',
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1))
                self.fm_channels_sum += layer.out_channels // 2

        # fully-connected layers for dimensionality reduction
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fm_channels_sum, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, D),
            nn.ReLU()
        )

        # final softmax classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(D, D // 2),
        #     nn.ReLU(),
        #     nn.Linear(D // 2, self.kwargs.get('n_classes', 2)),
        #     nn.Softmax(dim=1)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(D, self.kwargs.get('n_classes', 2)),
            nn.Softmax(dim=1)
        )

        # store conv feature maps
        self.feature_maps = None

        # spatial size of first feature map
        self.fm_size = None

    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            self.fm_size = (input_[0].size(2), input_[0].size(3))
            side_conv_name = 'side_conv0'
        else:
            side_conv_name = f'side_conv{self.feature_maps.size(0)}'

        output = getattr(self, side_conv_name)(output.clone())
        output = F.interpolate(output, self.fm_size,
                               mode='bilinear', align_corners=True)

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def forward(self, x):
        """Running a forward pass.

        Args:
            x: input image tensor of size (1, 3, H, W)

        Returns:
            pred: prediction with size (H, W, C)
        """

        height, width = x.size()[-2:]

        self.feature_maps = None
        _ = self.backbone(x)
        x = self.feature_maps
        x = x.view(x.size(0), -1)
        x = self.classifier(self.fc_layers(x.t()))

        return x.view(height, width, -1)


class WESUPTrainer(BaseTrainer):
    """Trainer for WESUP."""

    def __init__(self, model, **kwargs):
        """Initialize a WESUPTrainer instance.

        Kwargs:
            rescale_factor: rescale factor to subsample input images
            multiscale_range: multi-scale range for training
            class_weights: class weights for cross-entropy loss function
            sp_area: area of each superpixel
            sp_compactness: compactness parameter of SLIC
            enable_propagation: whether to enable label propagation
            propagate_threshold: threshold for label propagation
            propagate_weight: weight for label-propagated samples in loss function
            momentum: SGD momentum
            weight_decay: weight decay for optimizer
            freeze_backbone: whether to freeze backbone

        Returns:
            trainer: a new WESUPTrainer instance
        """

        config = WESUPConfig()
        if config.freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
        kwargs = {**config.to_dict(), **kwargs}
        super().__init__(model, **kwargs)

        self.kwargs = kwargs

        # cross-entropy loss function
        self.xentropy = partial(_cross_entropy)

        self.class_weights = self.kwargs.get('class_weights', (1, 1))
        self.mask_loss_weight = self.kwargs.get('mask_loss_weight', 1)
        self.relation_loss_weight = self.kwargs.get('relation_loss_weight', 1)
        self.propagate_weight = self.kwargs.get('propagate_weight', 1)

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            if osp.exists(osp.join(root_dir, 'points')):
                return Digest2019PointDataset(root_dir, proportion=proportion,
                                              multiscale_range=self.kwargs.get('multiscale_range'))
            return SegmentationDataset(root_dir, proportion=proportion,
                                       multiscale_range=self.kwargs.get('multiscale_range'))
        return SegmentationDataset(root_dir, rescale_factor=self.kwargs.get('rescale_factor'), train=False)

    def get_default_optimizer(self):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            # lr=5e-5,
            lr=1e-3,
            momentum=self.kwargs.get('momentum'),
            weight_decay=self.kwargs.get('weight_decay'),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)

        # return optimizer, None
        return optimizer, scheduler

    def preprocess(self, *data):
        data = [datum.to(self.device) for datum in data]
        if len(data) == 3:
            img, pixel_mask, point_mask = data
        elif len(data) == 2:
            img, pixel_mask = data
            point_mask = empty_tensor()
        elif len(data) == 1:
            img, = data
            point_mask = empty_tensor()
            pixel_mask = empty_tensor()
        else:
            raise ValueError('Invalid input data for WESUP')

        segments = slic(
            img.squeeze().cpu().numpy().transpose(1, 2, 0),
            n_segments=int(img.size(-2) * img.size(-1) /
                           self.kwargs.get('sp_area')),
            compactness=self.kwargs.get('sp_compactness'),
            start_label=0,
        )
        segments = torch.as_tensor(
            segments, dtype=torch.long, device=self.device)

        if point_mask is not None and not is_empty_tensor(point_mask):
            mask = point_mask.squeeze()
        elif pixel_mask is not None and not is_empty_tensor(pixel_mask):
            mask = pixel_mask.squeeze()
        else:
            mask = None

        sp_maps, sp_labels = _preprocess_superpixels(
            segments, mask, epsilon=self.kwargs.get('epsilon'))

        # return (img, sp_maps, sp_labels), (pixel_mask, sp_labels)
        return (img, sp_maps, sp_labels), (pixel_mask, sp_labels)

    def compute_loss(self, pred, target, metrics=None):
        pred, pseudo_labels, sp_relation_pred, sp_relation_target, \
            pseudo_relation_pred, pseudo_relation_target = pred

        if isinstance(self.class_weights, tuple):
            self.class_weights = torch.Tensor(
                self.class_weights).to(pred.device)

        mask_loss = self.xentropy(
            pred, pseudo_labels, self.class_weights, self.propagate_weight)

        relation_weights = None
        # relation_weights = self.class_weights
        if sp_relation_pred is not None:
            relation_loss = self.xentropy(
                sp_relation_pred, sp_relation_target, relation_weights)
        else:
            relation_loss = 0

        if pseudo_relation_pred is not None:
            pseudo_relation_loss = self.xentropy(
                pseudo_relation_pred, pseudo_relation_target, relation_weights)
        else:
            pseudo_relation_loss = 0

        loss = self.mask_loss_weight * mask_loss + \
            self.relation_loss_weight * relation_loss + \
            self.relation_loss_weight * 0.6 * pseudo_relation_loss

        # clear outdated prediction
        self.model.p_pred = None

        return loss

    def postprocess(self, pred, target=None):
        pred = pred[0].round().long()
        if target is not None:
            return pred, target[0].argmax(dim=1)
        return pred

    def post_epoch_hook(self, epoch):
        if self.scheduler is not None:
            labeled_loss = np.mean(self.tracker.history['loss'])

            # only adjust learning rate according to loss of labeled examples
            if 'propagate_loss' in self.tracker.history:
                labeled_loss -= np.mean(self.tracker.history['propagate_loss'])

            self.scheduler.step(labeled_loss)
