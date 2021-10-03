# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat, _tranpose_and_gather_feat_expand
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.post_process import ctdet_post_process

from cython_bbox import bbox_overlaps as bbox_ious

from .basetrack import BaseTrack, TrackState

import random
import pickle
import copy

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

gaussianBlurConv = GaussianBlurConv().cuda()

seed = 0
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Remove randomness (may be slower on Tesla GPUs)
# https://pytorch.org/docs/stable/notes/randomness.html
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
smoothL1 = torch.nn.SmoothL1Loss()
mse = torch.nn.MSELoss()



td_ = {}
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    shared_kalman_ = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.smooth_feat_ad = None

        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        self.det_dict = {}

    def update_features_ad(self, feat):
        feat /= np.linalg.norm(feat)
        if self.smooth_feat_ad is None:
            self.smooth_feat_ad = feat
        else:
            self.smooth_feat_ad = self.alpha * self.smooth_feat_ad + (1 - self.alpha) * feat
        self.smooth_feat_ad /= np.linalg.norm(self.smooth_feat_ad)

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_predict_(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman_.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def activate_(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id_()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def re_activate_(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id_()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.tracked_stracks_ad = []  # type: list[STrack]
        self.lost_stracks_ad = []  # type: list[STrack]
        self.removed_stracks_ad = []  # type: list[STrack]

        self.tracked_stracks_ = []  # type: list[STrack]
        self.lost_stracks_ = []  # type: list[STrack]
        self.removed_stracks_ = []  # type: list[STrack]

        self.frame_id = 0
        self.frame_id_ = 0
        self.frame_id_ad = 0

        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = 128

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.mean_ad = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std_ad = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.mean_ = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std_ = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()
        self.kalman_filter_ad = KalmanFilter()
        self.kalman_filter_ = KalmanFilter()

        self.attack_sg = True

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    @staticmethod
    def recoverImg(im_blob, img0):
        height = 608
        width = 1088
        im_blob = im_blob.cpu() * 255.0
        shape = img0.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        im_blob = im_blob.squeeze().permute(1, 2, 0)[top:height-bottom, left:width-right, :].numpy().astype(np.uint8)
        im_blob = cv2.cvtColor(im_blob, cv2.COLOR_RGB2BGR)

        h, w, _ = img0.shape
        im_blob = cv2.resize(im_blob, (w, h))

        return im_blob

    @staticmethod
    def insert_linear_pos(img_dt, resize, x_scale=None, y_scale=None):
        m_, n_ = img_dt.shape
        # 获取新的图像的大小
        if resize is None:
            n_new, m_new = np.round(x_scale * n_).astype(int), np.round(y_scale * m_).astype(int)
        else:
            n_new, m_new = resize

        n_scale, m_scale = n_ / n_new, m_ / m_new  # src_with/dst_with, Src_height/dst_heaight
        # 一、获取位置对应的四个点
        # 1-1- 初始化位置
        m_indxs = np.repeat(np.arange(m_new), n_new).reshape(m_new, n_new)
        n_indxs = np.array(list(range(n_new)) * m_new).reshape(m_new, n_new)
        # 1-2- 初始化位置
        m_indxs_c = (m_indxs + 0.5) * m_scale - 0.5
        n_indxs_c = (n_indxs + 0.5) * n_scale - 0.5
        ### 将小于零的数处理成0
        m_indxs_c[np.where(m_indxs_c < 0)] = 0.0
        n_indxs_c[np.where(n_indxs_c < 0)] = 0.0

        # 1-3 获取正方形顶点坐标
        m_indxs_c_down = m_indxs_c.astype(int)
        n_indxs_c_down = n_indxs_c.astype(int)
        m_indxs_c_up = m_indxs_c_down + 1
        n_indxs_c_up = n_indxs_c_down + 1
        ### 溢出部分修正
        m_max = m_ - 1
        n_max = n_ - 1
        m_indxs_c_up[np.where(m_indxs_c_up > m_max)] = m_max
        n_indxs_c_up[np.where(n_indxs_c_up > n_max)] = n_max

        # 1-4 获取正方形四个顶点的位置
        pos_0_0 = img_dt[m_indxs_c_down, n_indxs_c_down].astype(int)
        pos_0_1 = img_dt[m_indxs_c_up, n_indxs_c_down].astype(int)
        pos_1_1 = img_dt[m_indxs_c_up, n_indxs_c_up].astype(int)
        pos_1_0 = img_dt[m_indxs_c_down, n_indxs_c_up].astype(int)
        # 1-5 获取浮点位置
        m, n = np.modf(m_indxs_c)[0], np.modf(n_indxs_c)[0]
        return pos_0_0, pos_0_1, pos_1_1, pos_1_0, m, n

    def linear_insert_1color(self, img_dt, resize, fx=None, fy=None):
        pos_0_0, pos_0_1, pos_1_1, pos_1_0, m, n = self.insert_linear_pos(img_dt=img_dt, resize=resize, x_scale=fx,
                                                                     y_scale=fy)
        a = (pos_1_0 - pos_0_0)
        b = (pos_0_1 - pos_0_0)
        c = pos_1_1 + pos_0_0 - pos_1_0 - pos_0_1
        return np.round(a * n + b * m + c * n * m + pos_0_0).astype(int)

    def linear_insert(self, img_dt, resize, fx=None, fy=None):
        # 三个通道分开处理再合并
        if len(img_dt.shape) == 3:
            out_img0 = self.linear_insert_1color(img_dt[:, :, 0], resize=resize, fx=fx, fy=fy)
            out_img1 = self.linear_insert_1color(img_dt[:, :, 1], resize=resize, fx=fx, fy=fy)
            out_img2 = self.linear_insert_1color(img_dt[:, :, 2], resize=resize, fx=fx, fy=fy)
            out_img_all = np.c_[out_img0[:, :, np.newaxis], out_img1[:, :, np.newaxis], out_img2[:, :, np.newaxis]]
        else:
            out_img_all = self.linear_insert_1color(img_dt, resize=resize, fx=fx, fy=fy)
        return out_img_all.astype(np.int)

    # def recoverNoise(self, noise, img0):
    #     height = 608
    #     width = 1088
    #     noise = noise.cpu() * 255.0
    #     shape = img0.shape[:2]  # shape = [height, width]
    #     ratio = min(float(height) / shape[0], float(width) / shape[1])
    #     new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    #     dw = (width - new_shape[0]) / 2  # width padding
    #     dh = (height - new_shape[1]) / 2  # height padding
    #     top, bottom = round(dh - 0.1), round(dh + 0.1)
    #     left, right = round(dw - 0.1), round(dw + 0.1)
    #
    #     noise = noise.squeeze().permute(1, 2, 0)[top:height - bottom, left:width - right, :].numpy().astype(
    #         np.int)
    #     noise = noise[:, :, ::-1]
    #
    #     h, w, _ = img0.shape
    #     noise = self.linear_insert(noise, (w, h))
    #
    #     return noise

    def recoverNoise(self, noise, img0):
        height = 608
        width = 1088
        shape = img0.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        noise = noise[:, :, top:height - bottom, left:width - right]
        h, w, _ = img0.shape
        noise = self.resizeTensor(noise, h, w).cpu().squeeze().permute(1, 2, 0).numpy()

        noise = (noise[:, :, ::-1] * 255).astype(np.int)

        return noise

    def deRecoverNoise(self, noise, img0):
        height = 608
        width = 1088
        shape = img0.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        noise = noise[:, :, ::-1]
        noise = noise.astype(np.float) / 255
        noise = torch.from_numpy(noise).cuda().float()

        noise = noise.permute(2, 0, 1).unsqueeze(0)
        noise = self.resizeTensor(noise, height-bottom-top, width-right-left)

        noise_ = torch.zeros((1, 3, height, width)).cuda()

        noise_[:, :, top:height - bottom, left:width - right] = noise

        return noise_


    @staticmethod
    def resizeTensor(tensor, height, width):
        h = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width).to(tensor.device)
        w = torch.linspace(-1, 1, width).repeat(height, 1).to(tensor.device)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)

        output = F.grid_sample(tensor, grid=grid, mode='bilinear', align_corners=True)
        return output


    @staticmethod
    def processIoUs(ious):
        h, w = ious.shape
        assert h == w
        ious = np.tril(ious, -1)
        index = np.argsort(-ious.reshape(-1))
        indSet = set([])
        for ind in index:
            i = ind // h
            j = ind % w
            if ious[i, j] == 0:
                break
            if i in indSet or j in indSet:
                ious[i, j] = 0
            else:
                indSet.add(i)
                indSet.add(j)
        return ious

    def fgsm(self, im_blob, id_features, dets, epsilon=0.03):
        ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))

        ious = self.processIoUs(ious)
        loss = 0
        for id_feature in id_features:
            for i in range(dets.shape[0]):
                for j in range(i):
                    if ious[i, j] > 0:
                        loss += torch.mm(id_feature[i:i + 1], id_feature[j:j + 1].T).squeeze()
        if isinstance(loss, int):
            return torch.zeros_like(im_blob)
        loss.backward()
        noise = im_blob.grad.sign() * epsilon
        return noise


    def fgsmV1(self, im_blob, id_features, last_id_features, dets, epsilon=0.03):
        ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))

        ious = self.processIoUs(ious)
        loss = 0
        for id_feature in id_features:
            for i in range(dets.shape[0]):
                for j in range(i):
                    if ious[i, j] > 0:
                        if last_id_features[i] is not None:
                            last_id_feature = torch.from_numpy(last_id_features[i]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[i:i + 1], last_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[j:j + 1], last_id_feature.T).squeeze()
                        else:
                            loss += torch.mm(id_feature[i:i + 1], id_feature[j:j + 1].T).squeeze()

        if isinstance(loss, int):
            return torch.zeros_like(im_blob)
        loss.backward()
        noise = im_blob.grad.sign() * epsilon
        return noise


    def fgsmV2(self, im_blob, id_features, last_ad_id_features, dets, epsilon=0.03):
        ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))

        ious = self.processIoUs(ious)
        loss = 0
        for id_feature in id_features:
            for i in range(dets.shape[0]):
                for j in range(i):
                    if ious[i, j] > 0:
                        if last_ad_id_features[i] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[i]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                        if last_ad_id_features[j] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[j]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                        if last_ad_id_features[i] is None and last_ad_id_features[j] is None:
                            loss += torch.mm(id_feature[i:i + 1], id_feature[j:j + 1].T).squeeze()

        if isinstance(loss, int):
            return torch.zeros_like(im_blob)
        loss.backward()
        grad = im_blob.grad
        noise = grad.sign() * epsilon
        # noise = gaussianBlurConv(noise)
        return noise

    def fgsmV2_(self, im_blob, id_features, last_ad_id_features, dets, outputs_ori, outputs, epsilon=0.03):
        ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))

        ious = self.processIoUs(ious)
        loss = 0
        for id_feature in id_features:
            for i in range(dets.shape[0]):
                for j in range(i):
                    if ious[i, j] > 0:
                        if last_ad_id_features[i] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[i]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                        if last_ad_id_features[j] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[j]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                        if last_ad_id_features[i] is None and last_ad_id_features[j] is None:
                            loss += torch.mm(id_feature[i:i + 1], id_feature[j:j + 1].T).squeeze()

        loss_det = 0
        for key in ['hm', 'wh', 'reg']:
            loss_det -= smoothL1(outputs[key], outputs_ori[key].data)
        loss += loss_det
        if isinstance(loss, int):
            return torch.zeros_like(im_blob)
        loss.backward()
        grad = im_blob.grad
        noise = grad.sign() * epsilon
        # noise = gaussianBlurConv(noise)
        return noise

    def fgsm_l2(self, im_blob, id_features, last_ad_id_features, dets, outputs_ori=None, outputs=None, ori_im_blob=None, lr=0.1):
        ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))

        ious = self.processIoUs(ious)
        loss = 0
        for id_feature in id_features:
            for i in range(dets.shape[0]):
                for j in range(i):
                    if ious[i, j] > 0:
                        if last_ad_id_features[i] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[i]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                        if last_ad_id_features[j] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[j]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                        if last_ad_id_features[i] is None and last_ad_id_features[j] is None:
                            loss += torch.mm(id_feature[i:i + 1], id_feature[j:j + 1].T).squeeze()

        if ori_im_blob is not None:
            loss_det = 0
            for key in ['hm', 'wh', 'reg']:
                loss_det -= smoothL1(outputs[key], outputs_ori[key].data)
            loss += loss_det
            loss -= mse(im_blob, ori_im_blob)
        if isinstance(loss, int):
            return torch.zeros_like(im_blob)
        loss.backward()
        noise = im_blob.grad * lr
        # noise = gaussianBlurConv(noise)
        return noise

    def fgsm_l2_(self, im_blob, id_features, last_ad_id_features, dets, outputs_ori=None, outputs=None, ori_im_blob=None, lr=0.1):
        ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))

        ious = self.processIoUs(ious)
        loss = 0
        for id_feature in id_features:
            for i in range(dets.shape[0]):
                for j in range(i):
                    if ious[i, j] > 0:
                        if last_ad_id_features[i] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[i]).unsqueeze(0).cuda()
                            sim_1 = torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                            sim_2 = torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                            loss += min(sim_2 - sim_1, 0.2)
                        if last_ad_id_features[j] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[j]).unsqueeze(0).cuda()
                            sim_1 = torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                            sim_2 = torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                            loss += min(sim_2 - sim_1, 0.2)
                        if last_ad_id_features[i] is None and last_ad_id_features[j] is None:
                            loss += torch.mm(id_feature[i:i + 1], id_feature[j:j + 1].T).squeeze()
        if ori_im_blob is not None:
            loss_det = 0
            for key in ['hm', 'wh', 'reg']:
                loss_det -= smoothL1(outputs[key], outputs_ori[key].data)
            loss += loss_det
            loss -= mse(im_blob, ori_im_blob)
        if isinstance(loss, int):
            return torch.zeros_like(im_blob)
        loss.backward()
        noise = im_blob.grad * lr
        # noise = gaussianBlurConv(noise)
        return noise

    def fgsmV3(self, im_blob, id_features, last_id_features, last_ad_id_features, dets, epsilon=0.03):
        ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))

        ious = self.processIoUs(ious)
        loss = 0
        for id_feature in id_features:
            for i in range(dets.shape[0]):
                for j in range(i):
                    if ious[i, j] > 0:
                        if last_id_features[i] is not None:
                            last_id_feature = torch.from_numpy(last_id_features[i]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[i:i + 1], last_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[j:j + 1], last_id_feature.T).squeeze()
                        if last_ad_id_features[i] is not None:
                            last_ad_id_feature = torch.from_numpy(last_ad_id_features[i]).unsqueeze(0).cuda()
                            loss -= torch.mm(id_feature[i:i + 1], last_ad_id_feature.T).squeeze()
                            loss += torch.mm(id_feature[j:j + 1], last_ad_id_feature.T).squeeze()
                        if last_id_features[i] is None and last_ad_id_features[i] is None:
                            loss += torch.mm(id_feature[i:i + 1], id_feature[j:j + 1].T).squeeze()

        if isinstance(loss, int):
            return torch.zeros_like(im_blob)
        loss.backward()
        noise = im_blob.grad.sign() * epsilon
        return noise

    def ifgsmV2(self, im_blob, id_features, last_ad_id_features, dets, inds, remain_inds, outputs_ori, epsilon=0.03, alpha=0.003):
        noise = self.fgsmV2(im_blob, id_features, last_ad_id_features, dets, epsilon=alpha)
        num = int(epsilon / alpha)
        for i in range(num):
            im_blob_ad = torch.clip(im_blob + noise, min=0, max=1).data
            id_features, outputs = self.forwardFeature(im_blob_ad, inds, remain_inds)
            noise += self.fgsmV2_(im_blob_ad, id_features, last_ad_id_features, dets, outputs_ori, outputs, epsilon=alpha)
        return noise


    def ifgsm_gd(self, im_blob, id_features, last_ad_id_features, dets, inds, remain_inds, outputs_ori):
        noise = self.fgsm_l2_(im_blob, id_features, last_ad_id_features, dets)
        for i in range(50):
            im_blob_ad = torch.clip(im_blob + noise, min=0, max=1).data
            id_features, outputs = self.forwardFeature(im_blob_ad, inds, remain_inds)
            noise_ = self.fgsm_l2_(
                im_blob_ad, id_features, last_ad_id_features, dets,
                outputs_ori=outputs_ori,
                outputs=outputs,
                ori_im_blob=im_blob
            )
            noise += noise_
            if (noise_**2).sum().sqrt().item() < 1:
                break
        return noise

    def ifgsm_gd_sg(
            self,
            im_blob,
            img0,
            id_features,
            last_ad_id_features,
            dets,
            inds,
            remain_inds,
            last_info,
            outputs_ori,
            attack_id,
            attack_ind,
            target_id,
            target_ind,
            lr=0.1
    ):
        ae_id = attack_id
        noise = torch.zeros_like(im_blob)
        im_blob_ori = im_blob.clone().data
        hm_ori = outputs_ori['hm'].data * 2
        outputs = outputs_ori
        i = 0
        while ae_id == attack_id or ae_id is None:
            loss = 0
            for id_feature in id_features:
                #TODO last_ad_id_features改成攻击后的
                if last_ad_id_features[attack_ind] is not None:
                    last_ad_id_feature = torch.from_numpy(last_ad_id_features[attack_ind]).unsqueeze(0).cuda()
                    sim_1 = torch.mm(id_feature[attack_ind:attack_ind + 1], last_ad_id_feature.T).squeeze()
                    sim_2 = torch.mm(id_feature[target_ind:target_ind + 1], last_ad_id_feature.T).squeeze()
                    loss += sim_2 - sim_1
                if last_ad_id_features[target_ind] is not None:
                    last_ad_id_feature = torch.from_numpy(last_ad_id_features[target_ind]).unsqueeze(0).cuda()
                    sim_1 = torch.mm(id_feature[target_ind:target_ind + 1], last_ad_id_feature.T).squeeze()
                    sim_2 = torch.mm(id_feature[attack_ind:attack_ind + 1], last_ad_id_feature.T).squeeze()
                    loss += sim_2 - sim_1
                if last_ad_id_features[attack_ind] is None and last_ad_id_features[attack_ind] is None:
                    loss += torch.mm(id_feature[attack_ind:attack_ind + 1], id_feature[target_ind:target_ind + 1].T).squeeze()
            loss -= mse(im_blob, im_blob_ori)
            loss -= mse(outputs['hm'].view(-1)[inds[0][remain_inds]], hm_ori.view(-1)[inds[0][remain_inds]])

            loss.backward()
            noise += im_blob.grad * lr
            im_blob = torch.clip(im_blob_ori + noise, min=0, max=1).data
            id_features, last_ad_id_features_, outputs, ae_id = self.forwardFeatureSg(
                im_blob,
                img0,
                dets,
                inds,
                remain_inds,
                attack_id,
                attack_ind,
                target_id,
                target_ind,
                last_info
            )
            if last_ad_id_features_ is not None:
                last_ad_id_features = last_ad_id_features_
            i += 1
            if i > 15:
                print('fail')
                break
        return noise


    def forwardFeature(self, im_blob, inds, remain_inds):
        im_blob.requires_grad = True
        self.model.zero_grad()
        output = self.model(im_blob)[-1]
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)
        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0)
                id_features.append(id_feature_exp)
        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]
        return id_features, output

    def forwardFeatureSg(self, im_blob, img0, dets_, inds_, remain_inds_, attack_id, attack_ind, target_id, target_ind, last_info):
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        im_blob.requires_grad = True
        self.model.zero_grad()
        output = self.model(im_blob)[-1]
        hm = output['hm'].sigmoid()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg'] if self.opt.reg_offset else None
        dets_raw, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        dets = self.post_process(dets_raw.clone(), meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]

        ious = bbox_ious(np.ascontiguousarray(dets_[[attack_ind, target_ind], :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))
        det_ind = np.argmax(ious, axis=1)

        match = True
        if ious[0, det_ind[0]] < 0.9 or ious[1, det_ind[1]] < 0.9:
            dets = dets_
            inds = inds_
            remain_inds = remain_inds_
            match = False

        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0)
                id_features.append(id_feature_exp)

        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]

        ae_attack_id = None

        last_ad_id_features = None

        if not match:
            return id_features, None, output, ae_attack_id

        ae_attack_ind = det_ind[0]
        ae_target_ind = det_ind[1]

        index = list(range(len(id_features[0])))
        index[attack_ind] = ae_attack_ind
        index[target_ind] = ae_target_ind

        id_features_ = [torch.zeros_like(id_features[0]) for _ in range(len(id_features))]
        for i in range(9):
            id_features_[i] = id_features[i][index]
            id_features_[i] = id_features[i][index]

        id_feature = _tranpose_and_gather_feat_expand(id_feature, inds)
        id_feature = id_feature.squeeze(0)
        id_feature = id_feature[remain_inds]
        id_feature = id_feature.detach().cpu().numpy()

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        unconfirmed = copy.deepcopy(last_info['last_unconfirmed'])
        strack_pool = copy.deepcopy(last_info['last_strack_pool'])
        kalman_filter = copy.deepcopy(last_info['kalman_filter'])

        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
                # if self.frame_id_ == 34:
                #     import pdb; pdb.set_trace()
                return id_features_, last_ad_id_features, output, ae_attack_id

        ''' Step 3: Second association, with IOU'''
        for i, idet in enumerate(u_detection):
            if idet == ae_attack_ind:
                ae_attack_ind = i
                break
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
                # if self.frame_id_ == 34:
                #     import pdb; pdb.set_trace()
                return id_features_, last_ad_id_features, output, ae_attack_id


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        for i, idet in enumerate(u_detection):
            if idet == ae_attack_ind:
                ae_attack_ind = i
                break
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
                return id_features_, last_ad_id_features, output, ae_attack_id

        return id_features_, last_ad_id_features, output, ae_attack_id

    def CheckFit(self, dets, id_feature, attack_id, attack_ind, target_id, target_ind):
        attack_det = dets[attack_ind][:4]
        target_det = dets[target_ind][:4]
        ad_attack_det = None
        ad_target_det = None
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        unconfirmed = copy.deepcopy(self.ad_last_info['last_unconfirmed'])
        strack_pool = copy.deepcopy(self.ad_last_info['last_strack_pool'])
        kalman_filter = copy.deepcopy(self.ad_last_info['kalman_filter'])

        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.track_id == attack_id:
                ad_attack_det = det.tlbr
            elif track.track_id == target_id:
                ad_target_det = det.tlbr

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.track_id == attack_id:
                ad_attack_det = det.tlbr
            elif track.track_id == target_id:
                ad_target_det = det.tlbr

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            if track.track_id == attack_id:
                ad_attack_det = det.tlbr
            elif track.track_id == target_id:
                ad_target_det = det.tlbr

        if ad_attack_det is None or ad_target_det is None:
            return False

        ori_dets = np.array([attack_det, target_det])
        ad_dets = np.array([ad_attack_det, ad_target_det])

        ious = bbox_ious(ori_dets.astype(np.float), ad_dets.astype(np.float))
        if ious[0, 0] > 0.9 and ious[1, 1] > 0.9:
            # import pdb;
            # pdb.set_trace()
            return True
        return False


    def update_attack_(self, im_blob, img0, **kwargs):
        self.frame_id_ += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        # with torch.no_grad():
        im_blob.requires_grad = True
        self.model.zero_grad()
        output = self.model(im_blob)[-1]
        hm = output['hm'].sigmoid()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg'] if self.opt.reg_offset else None
        dets_raw, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0)
                id_features.append(id_feature_exp)

        id_feature = _tranpose_and_gather_feat_expand(id_feature, inds)

        id_feature = id_feature.squeeze(0)

        dets = self.post_process(dets_raw.clone(), meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]

        id_feature = id_feature.detach().cpu().numpy()

        last_id_features = [None for _ in range(len(dets))]
        last_ad_id_features = [None for _ in range(len(dets))]
        dets_index = [i for i in range(len(dets))]
        dets_ids = [None for _ in range(len(dets))]
        tracks_ad = []

        # import pdb; pdb.set_trace()
        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks_:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks_)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter_, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # import pdb; pdb.set_trace()
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = track.smooth_feat
            last_ad_id_features[dets_index[idet]] = track.smooth_feat_ad
            tracks_ad.append((track, dets_index[idet]))
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id_)
                activated_starcks.append(track)
            else:
                track.re_activate_(det, self.frame_id_, new_id=False)
                refind_stracks.append(track)
            dets_ids[dets_index[idet]] = track.track_id

        ''' Step 3: Second association, with IOU'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = track.smooth_feat
            last_ad_id_features[dets_index[idet]] = track.smooth_feat_ad
            tracks_ad.append((track, dets_index[idet]))
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id_)
                activated_starcks.append(track)
            else:
                track.re_activate_(det, self.frame_id_, new_id=False)
                refind_stracks.append(track)
            dets_ids[dets_index[idet]] = track.track_id

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = unconfirmed[itracked].smooth_feat
            last_ad_id_features[dets_index[idet]] = unconfirmed[itracked].smooth_feat_ad
            tracks_ad.append((unconfirmed[itracked], dets_index[idet]))
            unconfirmed[itracked].update(detections[idet], self.frame_id_)
            activated_starcks.append(unconfirmed[itracked])
            dets_ids[dets_index[idet]] = unconfirmed[itracked].track_id
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate_(self.kalman_filter_, self.frame_id_)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks_:
            if self.frame_id_ - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks_ = [t for t in self.tracked_stracks_ if t.state == TrackState.Tracked]
        self.tracked_stracks_ = joint_stracks(self.tracked_stracks_, activated_starcks)
        self.tracked_stracks_ = joint_stracks(self.tracked_stracks_, refind_stracks)
        self.lost_stracks_ = sub_stracks(self.lost_stracks_, self.tracked_stracks_)
        self.lost_stracks_.extend(lost_stracks)
        self.lost_stracks_ = sub_stracks(self.lost_stracks_, self.removed_stracks_)
        self.removed_stracks_.extend(removed_stracks)
        self.tracked_stracks_, self.lost_stracks_ = remove_duplicate_stracks(self.tracked_stracks_, self.lost_stracks_)
        # get scores of lost tracks
        output_stracks_ori = [track for track in self.tracked_stracks_ if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id_))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        attack = self.opt.attack
        if attack == 'fgsm':
            noise = self.fgsm(im_blob, id_features, dets)
        elif attack == 'fgsmV1':
            noise = self.fgsmV1(im_blob, id_features, last_id_features, dets)
        elif attack == 'fgsmV2':
            noise = self.fgsmV2(im_blob, id_features, last_ad_id_features, dets)
        elif attack == 'fgsmV3':
            noise = self.fgsmV3(im_blob, id_features, last_id_features, last_ad_id_features, dets)
        elif attack == 'ifgsmV2':
            noise = self.ifgsmV2(im_blob, id_features, last_ad_id_features, dets, inds, remain_inds, outputs_ori=output)
        elif attack == 'ifgsm_gd':
            noise = self.ifgsm_gd(im_blob, id_features, last_ad_id_features, dets, inds, remain_inds, outputs_ori=output)
        else:
            raise Exception(f'Cannot find {attack}')

        l2_dis = (noise ** 2).sum().sqrt().item()

        # noise = self.recoverNoise(noise, img0)
        # noise = self.deRecoverNoise(noise, img0)
        im_blob = torch.clip(im_blob+noise, min=0, max=1)
        # im_blob = (im_blob * 255).int().float() / 255

        self.update_ad(im_blob, img0, dets_raw, inds, tracks_ad, **kwargs)

        output_stracks_att = self.update(im_blob, img0, **kwargs)

        noise = self.recoverNoise(noise, img0)
        # import pdb; pdb.set_trace()
        adImg = np.clip(img0 + noise, a_min=0, a_max=255)

        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        noise = (noise * 255).astype(np.uint8)

        return output_stracks_ori, output_stracks_att, adImg, noise, l2_dis

    def update_attack(self, im_blob, img0, **kwargs):
        self.frame_id_ += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        # with torch.no_grad():
        im_blob.requires_grad = True
        self.model.zero_grad()
        output = self.model(im_blob)[-1]
        hm = output['hm'].sigmoid()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg'] if self.opt.reg_offset else None
        dets_raw, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0)
                id_features.append(id_feature_exp)

        id_feature = _tranpose_and_gather_feat_expand(id_feature, inds)

        id_feature = id_feature.squeeze(0)

        dets = self.post_process(dets_raw.clone(), meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]

        id_feature = id_feature.detach().cpu().numpy()

        last_id_features = [None for _ in range(len(dets))]
        last_ad_id_features = [None for _ in range(len(dets))]
        dets_index = [i for i in range(len(dets))]
        dets_ids = [None for _ in range(len(dets))]
        tracks_ad = []

        # import pdb; pdb.set_trace()
        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks_:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks_)

        last_strack_pool = copy.deepcopy(strack_pool)
        last_unconfirmed = copy.deepcopy(unconfirmed)

        last_info = {
            'last_strack_pool': last_strack_pool,
            'last_unconfirmed': last_unconfirmed,
            'kalman_filter': copy.deepcopy(self.kalman_filter_)
        }
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter_, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # import pdb; pdb.set_trace()
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = track.smooth_feat
            last_ad_id_features[dets_index[idet]] = track.smooth_feat_ad
            tracks_ad.append((track, dets_index[idet]))
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id_)
                activated_starcks.append(track)
            else:
                track.re_activate_(det, self.frame_id_, new_id=False)
                refind_stracks.append(track)
            dets_ids[dets_index[idet]] = track.track_id

        ''' Step 3: Second association, with IOU'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = track.smooth_feat
            last_ad_id_features[dets_index[idet]] = track.smooth_feat_ad
            tracks_ad.append((track, dets_index[idet]))
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id_)
                activated_starcks.append(track)
            else:
                track.re_activate_(det, self.frame_id_, new_id=False)
                refind_stracks.append(track)
            dets_ids[dets_index[idet]] = track.track_id

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = unconfirmed[itracked].smooth_feat
            last_ad_id_features[dets_index[idet]] = unconfirmed[itracked].smooth_feat_ad
            tracks_ad.append((unconfirmed[itracked], dets_index[idet]))
            unconfirmed[itracked].update(detections[idet], self.frame_id_)
            activated_starcks.append(unconfirmed[itracked])
            dets_ids[dets_index[idet]] = unconfirmed[itracked].track_id
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        dets_index = [dets_index[i] for i in u_detection]
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate_(self.kalman_filter_, self.frame_id_)
            activated_starcks.append(track)
            dets_ids[dets_index[inew]] = track.track_id
        """ Step 5: Update state"""
        for track in self.lost_stracks_:
            if self.frame_id_ - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks_ = [t for t in self.tracked_stracks_ if t.state == TrackState.Tracked]
        self.tracked_stracks_ = joint_stracks(self.tracked_stracks_, activated_starcks)
        self.tracked_stracks_ = joint_stracks(self.tracked_stracks_, refind_stracks)
        self.lost_stracks_ = sub_stracks(self.lost_stracks_, self.tracked_stracks_)
        self.lost_stracks_.extend(lost_stracks)
        self.lost_stracks_ = sub_stracks(self.lost_stracks_, self.removed_stracks_)
        self.removed_stracks_.extend(removed_stracks)
        self.tracked_stracks_, self.lost_stracks_ = remove_duplicate_stracks(self.tracked_stracks_, self.lost_stracks_)
        # get scores of lost tracks
        output_stracks_ori = [track for track in self.tracked_stracks_ if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id_))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        attack_id = 2

        attack = self.opt.attack

        noise = torch.zeros_like(im_blob)
        thr = 0.4
        if self.attack_sg and self.frame_id_ > 5:
            for attack_ind, track_id in enumerate(dets_ids):
                if track_id == attack_id:
                    ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                                     np.ascontiguousarray(dets[:, :4], dtype=np.float))

                    ious = self.processIoUs(ious)
                    ious = ious + ious.T
                    target_ind = np.argmax(ious[attack_ind])
                    if ious[attack_ind][target_ind] > thr:
                        target_id = dets_ids[target_ind]
                        fit = self.CheckFit(dets, id_feature, attack_id, attack_ind, target_id, target_ind)
                        if fit:
                            noise = self.ifgsm_gd_sg(
                                im_blob,
                                img0,
                                id_features,
                                last_ad_id_features,
                                dets,
                                inds,
                                remain_inds,
                                last_info=self.ad_last_info,
                                outputs_ori=output,
                                attack_id=attack_id,
                                attack_ind=attack_ind,
                                target_id=target_id,
                                target_ind=target_ind
                            )
                            thr = 0
                    else:
                        thr = 0.4
                        # self.attack_sg = False
                    break

        l2_dis = (noise ** 2).sum().sqrt().item()

        # noise = self.recoverNoise(noise, img0)
        # noise = self.deRecoverNoise(noise, img0)
        im_blob = torch.clip(im_blob+noise, min=0, max=1)
        # im_blob = (im_blob * 255).int().float() / 255

        self.update_ad(im_blob, img0, dets_raw, inds, tracks_ad, **kwargs)

        output_stracks_att = self.update(im_blob, img0, **kwargs)

        noise = self.recoverNoise(noise, img0)
        # import pdb; pdb.set_trace()
        adImg = np.clip(img0 + noise, a_min=0, a_max=255)

        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        noise = (noise * 255).astype(np.uint8)

        return output_stracks_ori, output_stracks_att, adImg, noise, l2_dis

    def update_ad(self, im_blob, img0, dets, inds, tracks_ad, **kwargs):
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            # hm = output['hm'].sigmoid_()
            # wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            # reg = output['reg'] if self.opt.reg_offset else None
            # dets, inds_ = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            # import pdb; pdb.set_trace()
            # id_feature = _tranpose_and_gather_feat(id_feature, inds)
            # id_feature = id_feature.squeeze(0)
            # id_feature = id_feature.detach().cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres


        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0)
                id_features.append(id_feature_exp)

        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]

        # import pdb; pdb.set_trace()
        for track, index in tracks_ad:
            lst = []
            for feat in id_features:
                lst.append(feat[index])
            feat = sum(lst)
            feat /= (feat ** 2).sum().sqrt()
            feat = feat.cpu().numpy()
            track.update_features_ad(feat)

    def update(self, im_blob, img0, **kwargs):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            id_feature_ = id_feature.permute(0, 2, 3, 1).view(-1, 512)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.detach().cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        # import pdb; pdb.set_trace()
        dets_index = inds[0][remain_inds].tolist()

        td = {}
        td_ind = {}
        dbg = False

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if dbg:
                td[track.track_id] = det.tlwh
                td_ind[track.track_id] = dets_index[idet]
                if track.track_id not in td_:
                    td_[track.track_id] = [None for i in range(50)]
                td_[track.track_id][self.frame_id] = (track.smooth_feat, det.smooth_feat)
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if dbg:
                td[track.track_id] = det.tlwh
                td_ind[track.track_id] = dets_index[idet]
                if track.track_id not in td_:
                    td_[track.track_id] = [None for i in range(50)]
                td_[track.track_id][self.frame_id] = (track.smooth_feat, det.smooth_feat)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            if dbg:
                td[unconfirmed[itracked].track_id] = detections[idet].tlwh
                td_ind[unconfirmed[itracked].track_id] = dets_index[idet]
                if unconfirmed[itracked].track_id not in td_:
                    td_[unconfirmed[itracked].track_id] = [None for i in range(50)]
                td_[unconfirmed[itracked].track_id][self.frame_id] = (unconfirmed[itracked].smooth_feat, detections[idet].smooth_feat)
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        if dbg:
            f_1 = []
            f_2 = []
            fi = 1
            fj = 2

            if self.frame_id == 25:
                for i in range(20):
                    if td_[fi][i] is None or td_[fj][i] is None:
                        continue
                    f_1.append(td_[fi][i][0] @ td_[fi][i][1])
                    f_2.append(td_[fi][i][0]@td_[fj][i][1])
                f1 = sum(f_1) / len(f_1)
                f2 = sum(f_2) / len(f_2)

                sc = 0
                for i in range(len(f_1)):
                    if f_2[i] > f_1[i]:
                        sc += 1
                print(f'f1:{f1}, f2:{f2}, sc:{sc}, len:{len(f_1)}')
                import pdb; pdb.set_trace()

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        self.ad_last_info = {
            'last_strack_pool': copy.deepcopy(strack_pool),
            'last_unconfirmed': copy.deepcopy(unconfirmed),
            'kalman_filter': copy.deepcopy(self.kalman_filter_)
        }

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def save(obj, name):
    with open(f'/home/derry/Desktop/{name}.pth', 'wb') as f:
        pickle.dump(obj, f)

def load(name):
    with open(f'/home/derry/Desktop/{name}.pth', 'rb') as f:
        obj = pickle.load(f)
    return obj