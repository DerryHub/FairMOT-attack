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

from scipy.optimize import linear_sum_assignment
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
        self.exist_len = 0

        self.smooth_feat = None
        self.smooth_feat_ad = None

        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        self.curr_tlbr = self.tlwh_to_tlbr(self._tlwh)

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

    def activate(self, kalman_filter, frame_id, track_id=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if track_id:
            self.track_id = track_id['track_id']
            track_id['track_id'] += 1
        else:
            self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def activate_(self, kalman_filter, frame_id, track_id=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if track_id:
            self.track_id = track_id['track_id']
            track_id['track_id'] += 1
        else:
            self.track_id = self.next_id_()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.exist_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def re_activate_(self, new_track, frame_id, new_id=False):
        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.exist_len += 1
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
        self.exist_len += 1

        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
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
    def __init__(
            self,
            opt,
            frame_rate=30,
            tracked_stracks=[],
            lost_stracks=[],
            removed_stracks=[],
            frame_id=0,
            ad_last_info={}
    ):
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

        self.log_index = []
        self.unconfirmed_ad_iou = None
        self.tracked_stracks_ad_iou = None
        self.strack_pool_ad_iou = None

        self.tracked_stracks = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.lost_stracks = copy.deepcopy(lost_stracks)  # type: list[STrack]
        self.removed_stracks = copy.deepcopy(removed_stracks)  # type: list[STrack]

        self.tracked_stracks_ad = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.lost_stracks_ad = copy.deepcopy(lost_stracks)  # type: list[STrack]
        self.removed_stracks_ad = copy.deepcopy(removed_stracks)  # type: list[STrack]

        self.tracked_stracks_ = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.lost_stracks_ = copy.deepcopy(lost_stracks)  # type: list[STrack]
        self.removed_stracks_ = copy.deepcopy(removed_stracks)  # type: list[STrack]

        self.frame_id = frame_id
        self.frame_id_ = frame_id
        self.frame_id_ad = frame_id

        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = 128

        # self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        # self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        #
        # self.mean_ad = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        # self.std_ad = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        #
        # self.mean_ = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        # self.std_ = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()
        self.kalman_filter_ad = KalmanFilter()
        self.kalman_filter_ = KalmanFilter()

        self.attack_sg = True
        self.attack_mt = True
        self.attacked_ids = set([])
        self.low_iou_ids = set([])
        self.ATTACK_IOU_THR = 0.3
        self.attack_iou_thr = self.ATTACK_IOU_THR
        self.ad_last_info = copy.deepcopy(ad_last_info)
        self.FRAME_THR = 10

        self.temp_i = 0

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

        im_blob = im_blob.squeeze().permute(1, 2, 0)[top:height - bottom, left:width - right, :].numpy().astype(
            np.uint8)
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
        noise = self.resizeTensor(noise, height - bottom - top, width - right - left)

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

    def fgsm_l2(self, im_blob, id_features, last_ad_id_features, dets, outputs_ori=None, outputs=None, ori_im_blob=None,
                lr=0.1):
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

    def fgsm_l2_(self, im_blob, id_features, last_ad_id_features, dets, outputs_ori=None, outputs=None,
                 ori_im_blob=None, lr=0.1):
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

    def ifgsmV2(self, im_blob, id_features, last_ad_id_features, dets, inds, remain_inds, outputs_ori, epsilon=0.03,
                alpha=0.003):
        noise = self.fgsmV2(im_blob, id_features, last_ad_id_features, dets, epsilon=alpha)
        num = int(epsilon / alpha)
        for i in range(num):
            im_blob_ad = torch.clip(im_blob + noise, min=0, max=1).data
            id_features, outputs = self.forwardFeature(im_blob_ad, inds, remain_inds)
            noise += self.fgsmV2_(im_blob_ad, id_features, last_ad_id_features, dets, outputs_ori, outputs,
                                  epsilon=alpha)
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
            if (noise_ ** 2).sum().sqrt().item() < 1:
                break
        return noise

    def ifgsm_gd_sg(
            self,
            im_blob,
            img0,
            id_features,
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
        img0_h, img0_w = img0.shape[:2]
        H, W = outputs_ori['hm'].size()[2:]
        r_w, r_h = img0_w / W, img0_h / H
        r_max = max(r_w, r_h)
        dh = H - min(r_w, r_h) * img0_h
        dw = W - min(r_w, r_h) * img0_w
        ae_id = attack_id
        noise = torch.zeros_like(im_blob)
        im_blob_ori = im_blob.clone().data
        hm_ori = outputs_ori['hm'].data * 2
        outputs = outputs_ori
        i = 0
        j = -1
        last_ad_id_features = [None for _ in range(len(id_features[0]))]
        strack_pool = copy.deepcopy(last_info['last_strack_pool'])
        last_attack_det = None
        last_target_det = None
        STrack.multi_predict(strack_pool)
        for strack in strack_pool:
            if strack.track_id == attack_id:
                last_ad_id_features[attack_ind] = strack.smooth_feat
                last_attack_det = torch.from_numpy(strack.tlbr).cuda()
                last_attack_det[[0, 2]] = (last_attack_det[[0, 2]] - 0.5 * W * (r_w - r_max)) / r_max
                last_attack_det[[1, 3]] = (last_attack_det[[1, 3]] - 0.5 * H * (r_h - r_max)) / r_max
            elif strack.track_id == target_id:
                last_ad_id_features[target_ind] = strack.smooth_feat
                last_target_det = torch.from_numpy(strack.tlbr).cuda()
                last_target_det[[0, 2]] = (last_target_det[[0, 2]] - 0.5 * W * (r_w - r_max)) / r_max
                last_target_det[[1, 3]] = (last_target_det[[1, 3]] - 0.5 * H * (r_h - r_max)) / r_max

        index = inds[0][remain_inds]
        ae_attack_ind, ae_target_ind = attack_ind, target_ind
        while True:
            loss = 0
            # for id_feature in id_features:
            id_feature = id_features[4]
            if last_ad_id_features[attack_ind] is not None:
                last_ad_id_feature = torch.from_numpy(last_ad_id_features[attack_ind]).unsqueeze(0).cuda()
                sim_1 = torch.mm(id_feature[attack_ind:attack_ind + 1], last_ad_id_feature.T).squeeze()
                sim_2 = torch.mm(id_feature[target_ind:target_ind + 1], last_ad_id_feature.T).squeeze()
                loss += sim_2 - sim_1  # min(sim_2 - sim_1, 0.2)
            if last_ad_id_features[target_ind] is not None:
                last_ad_id_feature = torch.from_numpy(last_ad_id_features[target_ind]).unsqueeze(0).cuda()
                sim_1 = torch.mm(id_feature[target_ind:target_ind + 1], last_ad_id_feature.T).squeeze()
                sim_2 = torch.mm(id_feature[attack_ind:attack_ind + 1], last_ad_id_feature.T).squeeze()
                loss += sim_2 - sim_1  # min(sim_2 - sim_1, 0.2)
            if last_ad_id_features[attack_ind] is None and last_ad_id_features[target_ind] is None:
                loss += torch.mm(id_feature[attack_ind:attack_ind + 1],
                                 id_feature[target_ind:target_ind + 1].T).squeeze()
            loss -= mse(im_blob, im_blob_ori)
            loss -= mse(outputs['hm'].view(-1)[inds[0][remain_inds]], hm_ori.view(-1)[inds[0][remain_inds]])

            if i >= 20:
                wh = outputs['wh'].permute(0, 2, 3, 1).view(-1, 2)[index]
                reg = outputs['reg'].permute(0, 2, 3, 1).view(-1, 2)[index]
                centers = torch.stack([index % W, index // W], dim=1)
                centers = centers + reg
                dets_output = torch.stack([
                    centers[:, 0] - wh[:, 0] / 2,
                    centers[:, 1] - wh[:, 1] / 2,
                    centers[:, 0] + wh[:, 0] / 2,
                    centers[:, 1] + wh[:, 1] / 2,
                ], dim=1)

                bboxs = torch.stack([dets_output[ae_attack_ind], dets_output[ae_target_ind]], dim=0)
                loss += self.bbox_iou_tensor(bboxs) * 0.5
                # import pdb; pdb.set_trace()

                # attack_det_center = (dets_output[ae_attack_ind][:2] + dets_output[ae_attack_ind][2:]) / 2
                # target_det_center = (dets_output[ae_target_ind][:2] + dets_output[ae_target_ind][2:]) / 2
                # iou_loss = 0
                # if last_attack_det is not None:
                #     att_lastAtt_det = torch.stack([dets_output[ae_attack_ind], last_attack_det])
                #     tar_lastAtt_det = torch.stack([dets_output[ae_target_ind], last_attack_det])
                #     iou_loss += self.bbox_iou_tensor(tar_lastAtt_det) - \
                #                 self.bbox_iou_tensor(att_lastAtt_det)
                #     # last_attack_det_center = (last_attack_det[:2] + last_attack_det[2:]) / 2
                #     # iou_loss += ((attack_det_center - last_attack_det_center)**2).sum() - \
                #     #             ((target_det_center - last_attack_det_center)**2).sum()
                # if last_target_det is not None:
                #     att_lastTar_det = torch.stack([dets_output[ae_attack_ind], last_target_det])
                #     tar_lastTar_det = torch.stack([dets_output[ae_target_ind], last_target_det])
                #     iou_loss += self.bbox_iou_tensor(att_lastTar_det) - \
                #                 self.bbox_iou_tensor(tar_lastTar_det)
                #     # last_target_det_center = (last_target_det[:2] + last_target_det[2:]) / 2
                #     # iou_loss += ((target_det_center - last_target_det_center) ** 2).sum() - \
                #     #             ((attack_det_center - last_target_det_center) ** 2).sum()
                # loss += iou_loss * 0.5

                # import pdb; pdb.set_trace()

            loss.backward()
            noise += im_blob.grad * lr
            im_blob = torch.clip(im_blob_ori + noise, min=0, max=1).data
            id_features, last_ad_id_features_, outputs, ae_id, index, ae_attack_ind, ae_target_ind = self.forwardFeatureSg(
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
            if ae_id != attack_id:
                if j == -1:
                    j = 0
                elif j == 3:
                    break
                else:
                    j += 1

            if last_ad_id_features_ is not None:
                last_ad_id_features = last_ad_id_features_
            i += 1
            if i > 50:
                return None
                # break
        print(i)
        return noise

    def ifgsm_adam_sg(
            self,
            im_blob,
            img0,
            id_features,
            dets,
            inds,
            remain_inds,
            last_info,
            outputs_ori,
            attack_id,
            attack_ind,
            target_id,
            target_ind,
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999
    ):
        img0_h, img0_w = img0.shape[:2]
        H, W = outputs_ori['hm'].size()[2:]
        r_w, r_h = img0_w / W, img0_h / H
        r_max = max(r_w, r_h)
        noise = torch.zeros_like(im_blob)
        im_blob_ori = im_blob.clone().data
        outputs = outputs_ori
        wh_ori = outputs['wh'].clone().data
        reg_ori = outputs['reg'].clone().data

        last_ad_id_features = [None for _ in range(len(id_features[0]))]
        strack_pool = copy.deepcopy(last_info['last_strack_pool'])
        last_attack_det = None
        last_target_det = None
        STrack.multi_predict(strack_pool)
        for strack in strack_pool:
            if strack.track_id == attack_id:
                last_ad_id_features[attack_ind] = strack.smooth_feat
                last_attack_det = torch.from_numpy(strack.tlbr).cuda().float()
                last_attack_det[[0, 2]] = (last_attack_det[[0, 2]] - 0.5 * W * (r_w - r_max)) / r_max
                last_attack_det[[1, 3]] = (last_attack_det[[1, 3]] - 0.5 * H * (r_h - r_max)) / r_max
            elif strack.track_id == target_id:
                last_ad_id_features[target_ind] = strack.smooth_feat
                last_target_det = torch.from_numpy(strack.tlbr).cuda().float()
                last_target_det[[0, 2]] = (last_target_det[[0, 2]] - 0.5 * W * (r_w - r_max)) / r_max
                last_target_det[[1, 3]] = (last_target_det[[1, 3]] - 0.5 * H * (r_h - r_max)) / r_max
        last_attack_det_center = torch.round(
            (last_attack_det[:2] + last_attack_det[2:]) / 2) if last_attack_det is not None else None
        last_target_det_center = torch.round(
            (last_target_det[:2] + last_target_det[2:]) / 2) if last_target_det is not None else None

        hm_index = inds[0][remain_inds]
        hm_index_ori = copy.deepcopy(hm_index)

        for i in range(len(id_features)):
            id_features[i] = id_features[i][[attack_ind, target_ind]]

        adam_m = 0
        adam_v = 0

        i = 0
        j = -1
        suc = True
        while True:
            i += 1
            loss = 0
            loss_feat = 0
            for id_i, id_feature in enumerate(id_features):
                # id_feature = id_features[4]
                if last_ad_id_features[attack_ind] is not None:
                    last_ad_id_feature = torch.from_numpy(last_ad_id_features[attack_ind]).unsqueeze(0).cuda()
                    sim_1 = torch.mm(id_feature[0:0 + 1], last_ad_id_feature.T).squeeze()
                    sim_2 = torch.mm(id_feature[1:1 + 1], last_ad_id_feature.T).squeeze()
                    loss_feat += sim_2 - sim_1
                if last_ad_id_features[target_ind] is not None:
                    last_ad_id_feature = torch.from_numpy(last_ad_id_features[target_ind]).unsqueeze(0).cuda()
                    sim_1 = torch.mm(id_feature[1:1 + 1], last_ad_id_feature.T).squeeze()
                    sim_2 = torch.mm(id_feature[0:0 + 1], last_ad_id_feature.T).squeeze()
                    loss_feat += sim_2 - sim_1
                if last_ad_id_features[attack_ind] is None and last_ad_id_features[target_ind] is None:
                    loss_feat += torch.mm(id_feature[0:0 + 1], id_feature[1:1 + 1].T).squeeze()
            loss += loss_feat / len(id_features)
            loss -= mse(im_blob, im_blob_ori)

            if i in [10, 20, 30, 35, 40, 45, 50, 55]:
                attack_det_center = torch.stack([hm_index[attack_ind] % W, hm_index[attack_ind] // W]).float()
                target_det_center = torch.stack([hm_index[target_ind] % W, hm_index[target_ind] // W]).float()
                if last_target_det_center is not None:
                    attack_center_delta = attack_det_center - last_target_det_center
                    if torch.max(torch.abs(attack_center_delta)) > 2:
                        attack_center_delta /= torch.max(torch.abs(attack_center_delta))
                        attack_det_center = torch.round(attack_det_center - attack_center_delta).int()
                        hm_index[attack_ind] = attack_det_center[0] + attack_det_center[1] * W
                if last_attack_det_center is not None:
                    target_center_delta = target_det_center - last_attack_det_center
                    if torch.max(torch.abs(target_center_delta)) > 2:
                        target_center_delta /= torch.max(torch.abs(target_center_delta))
                        target_det_center = torch.round(target_det_center - target_center_delta).int()
                        hm_index[target_ind] = target_det_center[0] + target_det_center[1] * W
            # if i == 40:
            #     lr /= 10

            loss += ((1 - outputs['hm'].view(-1).sigmoid()[hm_index]) ** 2 *
                     torch.log(outputs['hm'].view(-1).sigmoid()[hm_index])).mean()

            loss -= mse(outputs['wh'].view(-1)[hm_index], wh_ori.view(-1)[hm_index_ori])
            loss -= mse(outputs['reg'].view(-1)[hm_index], reg_ori.view(-1)[hm_index_ori])

            loss.backward()

            grad = im_blob.grad

            adam_m = beta_1 * adam_m + (1 - beta_1) * grad
            adam_v = beta_2 * adam_v + (1 - beta_2) * (grad ** 2)

            adam_m_ = adam_m / (1 - beta_1 ** i)
            adam_v_ = adam_v / (1 - beta_2 ** i)

            update_grad = lr * adam_m_ / (adam_v_.sqrt() + 1e-8)

            noise += update_grad

            im_blob = torch.clip(im_blob_ori + noise, min=0, max=1).data
            id_features_, outputs_, ae_id = self.forwardFeatureSg(
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
            if id_features_ is not None:
                id_features = id_features_
            if outputs_ is not None:
                outputs = outputs_
            if ae_id != attack_id and ae_id is not None:
                break

            if i > 80:
                suc = False
                break
                # return None, -1
        return noise, i, suc

    def ifgsm_adam_mt(
            self,
            im_blob,
            img0,
            id_features,
            dets,
            inds,
            remain_inds,
            last_info,
            outputs_ori,
            attack_ids,
            attack_inds,
            target_ids,
            target_inds,
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999
    ):
        img0_h, img0_w = img0.shape[:2]
        H, W = outputs_ori['hm'].size()[2:]
        r_w, r_h = img0_w / W, img0_h / H
        r_max = max(r_w, r_h)
        noise = torch.zeros_like(im_blob)
        im_blob_ori = im_blob.clone().data
        outputs = outputs_ori
        i = 0
        j = -1
        last_ad_id_features = [None for _ in range(len(id_features[0]))]
        strack_pool = copy.deepcopy(last_info['last_strack_pool'])
        last_attack_dets = [None] * len(attack_ids)
        last_target_dets = [None] * len(target_ids)
        STrack.multi_predict(strack_pool)
        for strack in strack_pool:
            if strack.track_id in attack_ids:
                index = attack_ids.tolist().index(strack.track_id)
                last_ad_id_features[attack_inds[index]] = strack.smooth_feat
                last_attack_dets[index] = torch.from_numpy(strack.tlbr).cuda().float()
                last_attack_dets[index][[0, 2]] = (last_attack_dets[index][[0, 2]] - 0.5 * W * (r_w - r_max)) / r_max
                last_attack_dets[index][[1, 3]] = (last_attack_dets[index][[1, 3]] - 0.5 * H * (r_h - r_max)) / r_max
            if strack.track_id in target_ids:
                index = target_ids.tolist().index(strack.track_id)
                last_ad_id_features[target_inds[index]] = strack.smooth_feat
                last_target_dets[index] = torch.from_numpy(strack.tlbr).cuda().float()
                last_target_dets[index][[0, 2]] = (last_target_dets[index][[0, 2]] - 0.5 * W * (r_w - r_max)) / r_max
                last_target_dets[index][[1, 3]] = (last_target_dets[index][[1, 3]] - 0.5 * H * (r_h - r_max)) / r_max

        last_attack_dets_center = []
        for det in last_attack_dets:
            if det is None:
                last_attack_dets_center.append(None)
            else:
                last_attack_dets_center.append((det[:2] + det[2:]) / 2)
        last_target_dets_center = []
        for det in last_target_dets:
            if det is None:
                last_target_dets_center.append(None)
            else:
                last_target_dets_center.append((det[:2] + det[2:]) / 2)

        hm_index = inds[0][remain_inds]

        adam_m = 0
        adam_v = 0
        while True:
            i += 1
            loss = 0
            loss_feat = 0
            for index, attack_id in enumerate(attack_ids):
                target_id = target_ids[index]
                attack_ind = attack_inds[index]
                target_ind = target_inds[index]
                for id_i, id_feature in enumerate(id_features):
                    if last_ad_id_features[attack_ind] is not None:
                        last_ad_id_feature = torch.from_numpy(last_ad_id_features[attack_ind]).unsqueeze(0).cuda()
                        sim_1 = torch.mm(id_feature[attack_ind:attack_ind + 1], last_ad_id_feature.T).squeeze()
                        sim_2 = torch.mm(id_feature[target_ind:target_ind + 1], last_ad_id_feature.T).squeeze()
                        loss_feat += sim_2 - sim_1
                    if last_ad_id_features[target_ind] is not None:
                        last_ad_id_feature = torch.from_numpy(last_ad_id_features[target_ind]).unsqueeze(0).cuda()
                        sim_1 = torch.mm(id_feature[target_ind:target_ind + 1], last_ad_id_feature.T).squeeze()
                        sim_2 = torch.mm(id_feature[attack_ind:attack_ind + 1], last_ad_id_feature.T).squeeze()
                        loss_feat += sim_2 - sim_1
                    if last_ad_id_features[attack_ind] is None and last_ad_id_features[target_ind] is None:
                        loss_feat += torch.mm(id_feature[attack_ind:attack_ind + 1],
                                              id_feature[target_ind:target_ind + 1].T).squeeze()

                if i in [10, 20, 30, 40]:
                    attack_det_center = torch.stack([hm_index[attack_ind] % W, hm_index[attack_ind] // W]).float()
                    target_det_center = torch.stack([hm_index[target_ind] % W, hm_index[target_ind] // W]).float()
                    if last_target_dets_center[index] is not None:
                        attack_center_delta = attack_det_center - last_target_dets_center[index]
                        if torch.max(torch.abs(attack_center_delta)) > 2:
                            attack_center_delta /= torch.max(torch.abs(attack_center_delta))
                            attack_det_center = torch.round(attack_det_center - attack_center_delta).int()
                            hm_index[attack_ind] = attack_det_center[0] + attack_det_center[1] * W
                    if last_attack_dets_center[index] is not None:
                        target_center_delta = target_det_center - last_attack_dets_center[index]
                        if torch.max(torch.abs(target_center_delta)) > 2:
                            target_center_delta /= torch.max(torch.abs(target_center_delta))
                            target_det_center = torch.round(target_det_center - target_center_delta).int()
                            hm_index[target_ind] = target_det_center[0] + target_det_center[1] * W

            loss += loss_feat / len(id_features)
            loss -= mse(im_blob, im_blob_ori)

            loss += ((1 - outputs['hm'].view(-1).sigmoid()[hm_index]) ** 2 *
                     torch.log(outputs['hm'].view(-1).sigmoid()[hm_index])).mean()

            loss.backward()

            grad = im_blob.grad

            adam_m = beta_1 * adam_m + (1 - beta_1) * grad
            adam_v = beta_2 * adam_v + (1 - beta_2) * (grad ** 2)

            adam_m_ = adam_m / (1 - beta_1 ** i)
            adam_v_ = adam_v / (1 - beta_2 ** i)

            update_grad = lr * adam_m_ / (adam_v_.sqrt() + 1e-8)

            noise += update_grad

            im_blob = torch.clip(im_blob_ori + noise, min=0, max=1).data
            id_features, outputs, fail_ids = self.forwardFeatureMt(
                im_blob,
                img0,
                dets,
                inds,
                remain_inds,
                attack_ids,
                attack_inds,
                target_ids,
                target_inds,
                last_info
            )
            if fail_ids == 0:
                break
                # if j == -1:
                #     j = 0
                # elif j == 3:
                #     break
                # else:
                #     j += 1
            # print(torch.cat([index.view(-1, 1) // W, index.view(-1, 1) % W], dim=1))
            if i > 50:
                return None, -1
        return noise, i

    def bbox_iou_tensor(self, bboxs):
        xy1 = torch.max(bboxs[:, :2], dim=0)[0]
        xy2 = torch.min(bboxs[:, 2:], dim=0)[0]
        inter = torch.prod(xy2 - xy1)
        iou = inter / (torch.prod(bboxs[:, 2:] - bboxs[:, :2], dim=1).sum() - inter + 1)
        return iou

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

    def forwardFeatureSg(self, im_blob, img0, dets_, inds_, remain_inds_, attack_id, attack_ind, target_id, target_ind,
                         last_info):
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
        if ious[0, det_ind[0]] < 0.6 or ious[1, det_ind[1]] < 0.6:
            dets = dets_
            inds = inds_
            remain_inds = remain_inds_
            match = False
        # assert match
        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0)
                id_features.append(id_feature_exp)

        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]

        ae_attack_id = None

        if not match:
            for i in range(len(id_features)):
                id_features[i] = id_features[i][[attack_ind, target_ind]]
            return id_features, output, ae_attack_id

        ae_attack_ind = det_ind[0]
        ae_target_ind = det_ind[1]

        id_features_ = [None for _ in range(len(id_features))]
        for i in range(len(id_features)):
            id_features_[i] = id_features[i][[ae_attack_ind, ae_target_ind]]

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
                return id_features_, output, ae_attack_id

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
                return id_features_, output, ae_attack_id

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
                return id_features_, output, ae_attack_id

        return id_features_, output, ae_attack_id

    def forwardFeatureMt(self, im_blob, img0, dets_, inds_, remain_inds_, attack_ids, attack_inds, target_ids,
                         target_inds, last_info):
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
        dets_index = [i for i in range(len(dets))]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]

        ious = bbox_ious(np.ascontiguousarray(dets_[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))

        row_inds, col_inds = linear_sum_assignment(-ious)

        match = True

        for index, attack_ind in enumerate(attack_inds):
            target_ind = target_inds[index]
            if attack_ind not in row_inds or target_ind not in row_inds:
                match = False
                break
            att_index = row_inds.tolist().index(attack_ind)
            tar_index = row_inds.tolist().index(target_ind)
            if ious[attack_ind, col_inds[att_index]] < 0.6 or ious[target_ind, col_inds[tar_index]] < 0.6:
                match = False
                break
        if not match:
            dets = dets_
            inds = inds_
            remain_inds = remain_inds_
        # assert match
        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0)
                id_features.append(id_feature_exp)

        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]

        fail_ids = 0

        if not match:
            return id_features, output, None

        ae_attack_inds = []
        for attack_ind in attack_inds:
            ae_attack_inds.append(col_inds[row_inds == attack_ind])
        ae_target_inds = []
        for target_ind in target_inds:
            ae_target_inds.append(col_inds[row_inds == target_ind])

        ae_attack_inds = np.concatenate(ae_attack_inds)
        ae_target_inds = np.concatenate(ae_target_inds)

        id_features_ = [torch.zeros([len(dets_), id_features[0].size(1)]).to(id_features[0].device) for _ in range(len(id_features))]
        for i in range(9):
            id_features_[i][row_inds] = id_features[i][col_inds]

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
            if dets_index[idet] in ae_attack_inds:
                index = ae_attack_inds.tolist().index(dets_index[idet])
                if track.track_id == attack_ids[index]:
                    fail_ids += 1

        ''' Step 3: Second association, with IOU'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if dets_index[idet] in ae_attack_inds:
                index = ae_attack_inds.tolist().index(dets_index[idet])
                if track.track_id == attack_ids[index]:
                    fail_ids += 1

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            if dets_index[idet] in ae_attack_inds:
                index = ae_attack_inds.tolist().index(dets_index[idet])
                if track.track_id == attack_ids[index]:
                    fail_ids += 1

        return id_features_, output, fail_ids

    def CheckFit_(self, dets, id_feature, attack_id, attack_ind):
        attack_det = dets[attack_ind][:4]
        ad_attack_det = None
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

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            if track.track_id == attack_id:
                ad_attack_det = det.tlbr

        if ad_attack_det is None:
            return False

        ori_dets = np.array([attack_det])
        ad_dets = np.array([ad_attack_det])

        ious = bbox_ious(ori_dets.astype(np.float), ad_dets.astype(np.float))
        if ious[0, 0] > 0.9:
            return True
        return False

    def CheckFit(self, dets, id_feature, attack_ids, attack_inds):
        attack_dets = dets[attack_inds][:4]
        ad_attack_dets = []
        ad_attack_ids = []
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
            if track.track_id in attack_ids:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.track_id in attack_ids:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            if track.track_id in attack_ids:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        if len(ad_attack_dets) == 0:
            return []

        ori_dets = np.array(attack_dets)
        ad_dets = np.array(ad_attack_dets)

        ious = bbox_ious(ori_dets.astype(np.float), ad_dets.astype(np.float))
        ious_inds = np.argmax(ious, axis=1)
        attack_index = []
        for i, ind in enumerate(ious_inds):
            if ious[i, ind] > 0.9:
                attack_index.append(i)

        return attack_index

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
        # for strack in strack_pool:
        # strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
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
            noise = self.ifgsm_gd(im_blob, id_features, last_ad_id_features, dets, inds, remain_inds,
                                  outputs_ori=output)
        else:
            raise Exception(f'Cannot find {attack}')

        l2_dis = (noise ** 2).sum().sqrt().item()

        # noise = self.recoverNoise(noise, img0)
        # noise = self.deRecoverNoise(noise, img0)
        im_blob = torch.clip(im_blob + noise, min=0, max=1)
        # im_blob = (im_blob * 255).int().float() / 255

        self.update_ad(im_blob, img0, dets_raw, inds, tracks_ad, **kwargs)

        output_stracks_att = self.update(im_blob, img0, **kwargs)

        noise = self.recoverNoise(noise, img0)
        # import pdb; pdb.set_trace()
        adImg = np.clip(img0 + noise, a_min=0, a_max=255)

        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        noise = (noise * 255).astype(np.uint8)

        return output_stracks_ori, output_stracks_att, adImg, noise, l2_dis

    def update_attack_sg(self, im_blob, img0, **kwargs):
        self.frame_id_ += 1
        attack_id = kwargs['attack_id']
        self_track_id_ori = kwargs.get('track_id', {}).get('origin', None)
        self_track_id_att = kwargs.get('track_id', {}).get('attack', None)
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

        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
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
            track.activate_(self.kalman_filter_, self.frame_id_, track_id=self_track_id_ori)
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

        attack = self.opt.attack
        noise = None
        suc = 0
        if self.attack_sg:
            for attack_ind, track_id in enumerate(dets_ids):
                if track_id == attack_id:
                    if self.opt.attack_id > 0:
                        if not hasattr(self, f'frames_{attack_id}'):
                            setattr(self, f'frames_{attack_id}', 0)
                        if getattr(self, f'frames_{attack_id}') < self.FRAME_THR:
                            setattr(self, f'frames_{attack_id}', getattr(self, f'frames_{attack_id}') + 1)
                            break
                    ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float64),
                                     np.ascontiguousarray(dets[:, :4], dtype=np.float64))

                    ious = self.processIoUs(ious)
                    ious = ious + ious.T
                    target_ind = np.argmax(ious[attack_ind])
                    if ious[attack_ind][target_ind] >= self.attack_iou_thr:
                        target_id = dets_ids[target_ind]
                        fit = self.CheckFit(dets, id_feature, [attack_id], [attack_ind])
                        if fit:
                            noise, attack_iter, suc = self.ifgsm_adam_sg(
                                im_blob,
                                img0,
                                id_features,
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
                            self.attack_iou_thr = 0
                            if suc:
                                suc = 1
                                print(
                                    f'attack id: {attack_id}\tattack frame {self.frame_id_}: SUCCESS\tl2 distance: {(noise ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                            else:
                                suc = 2
                                print(
                                    f'attack id: {attack_id}\tattack frame {self.frame_id_}: FAIL\tl2 distance: {(noise ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                        elif ious[attack_ind][target_ind] > self.attack_iou_thr and self.CheckFit(dets, id_feature, [target_id], [target_ind]):
                            noise, attack_iter, suc = self.ifgsm_adam_sg(
                                im_blob,
                                img0,
                                id_features,
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
                            self.attack_iou_thr = 0
                            if suc:
                                suc = 1
                                print(
                                    f'attack id: {attack_id}\tattack frame {self.frame_id_}: SUCCESS\tl2 distance: {(noise ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                            else:
                                suc = 2
                                print(
                                    f'attack id: {attack_id}\tattack frame {self.frame_id_}: FAIL\tl2 distance: {(noise ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                        else:
                            suc = 3
                        if ious[attack_ind][target_ind] == 0:
                            self.temp_i += 1
                            if self.temp_i >= 10:
                                self.attack_iou_thr = self.ATTACK_IOU_THR
                        else:
                            self.temp_i = 0
                    else:
                        self.attack_iou_thr = self.ATTACK_IOU_THR
                    break

        if noise is not None:
            l2_dis = (noise ** 2).sum().sqrt().item()
            im_blob = torch.clip(im_blob + noise, min=0, max=1)

            noise = self.recoverNoise(noise, img0)
            adImg = np.clip(img0 + noise, a_min=0, a_max=255)

            noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
            noise = (noise * 255).astype(np.uint8)
        else:
            l2_dis = None
            adImg = img0
        output_stracks_att = self.update(im_blob, img0, track_id=self_track_id_att)

        return output_stracks_ori, output_stracks_att, adImg, noise, l2_dis, suc

    def update_attack_mt(self, im_blob, img0, **kwargs):
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
        ad_dets_index = [i for i in range(len(dets))]
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

        ad_unconfirmed = copy.deepcopy(self.ad_last_info['last_unconfirmed']) if self.ad_last_info else None
        ad_strack_pool = copy.deepcopy(self.ad_last_info['last_strack_pool']) if self.ad_last_info else None

        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
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
            if track.exist_len > self.FRAME_THR:
                dets_ids[dets_index[idet]] = track.track_id

        if ad_strack_pool:
            STrack.multi_predict(ad_strack_pool)
            dists = matching.embedding_distance(ad_strack_pool, detections)
            dists = matching.fuse_motion(self.kalman_filter_, dists, ad_strack_pool, detections)
            ad_matches, ad_u_track, ad_u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in ad_matches:
                track = ad_strack_pool[itracked]
                if dets_ids[ad_dets_index[idet]] not in self.attacked_ids and track.exist_len > self.FRAME_THR:
                    dets_ids[ad_dets_index[idet]] = track.track_id

        ''' Step 3: Second association, with IOU'''
        dets_index = [dets_index[i] for i in u_detection]
        detections_ = detections
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
            if track.exist_len > self.FRAME_THR:
                dets_ids[dets_index[idet]] = track.track_id

        if ad_strack_pool:
            ad_dets_index = [ad_dets_index[i] for i in ad_u_detection]
            ad_detections = [detections_[i] for i in ad_u_detection]
            ad_r_tracked_stracks = [ad_strack_pool[i] for i in ad_u_track if
                                    ad_strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(ad_r_tracked_stracks, ad_detections)
            ad_matches, ad_u_track, ad_u_detection = matching.linear_assignment(dists, thresh=0.5)
            for itracked, idet in ad_matches:
                track = ad_strack_pool[itracked]
                if dets_ids[ad_dets_index[idet]] not in self.attacked_ids and track.exist_len > self.FRAME_THR:
                    dets_ids[ad_dets_index[idet]] = track.track_id

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
            if track.exist_len > self.FRAME_THR:
                dets_ids[dets_index[idet]] = unconfirmed[itracked].track_id

        if ad_strack_pool:
            ad_dets_index = [ad_dets_index[i] for i in ad_u_detection]
            ad_detections = [ad_detections[i] for i in ad_u_detection]
            dists = matching.iou_distance(ad_unconfirmed, ad_detections)
            ad_matches, ad_u_unconfirmed, ad_u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in ad_matches:
                track = ad_strack_pool[itracked]
                if dets_ids[ad_dets_index[idet]] not in self.attacked_ids and track.exist_len > self.FRAME_THR:
                    dets_ids[ad_dets_index[idet]] = track.track_id

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
            if track.exist_len > self.FRAME_THR:
                dets_ids[dets_index[inew]] = track.track_id

        if ad_strack_pool:
            ad_dets_index = [ad_dets_index[i] for i in ad_u_detection]
            for inew in ad_u_detection:
                track = ad_detections[inew]
                if dets_ids[ad_dets_index[inew]] not in self.attacked_ids and track.exist_len > self.FRAME_THR:
                    dets_ids[ad_dets_index[inew]] = track.track_id
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

        attack_ids = []
        target_ids = []
        attack_inds = []
        target_inds = []

        attack = self.opt.attack
        noise = torch.zeros_like(im_blob)
        if self.attack_mt and self.frame_id_ > 20 and len(dets) > 0:
            ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                             np.ascontiguousarray(dets[:, :4], dtype=np.float))
            ious[range(len(dets)), range(len(dets))] = 0
            ious_inds = np.argmax(ious, axis=1)
            for attack_ind, track_id in enumerate(dets_ids):
                if ious[attack_ind, ious_inds[attack_ind]] > self.ATTACK_IOU_THR or (
                        track_id in self.low_iou_ids and ious[attack_ind, ious_inds[attack_ind]] > 0
                ):
                    attack_ids.append(track_id)
                    target_ids.append(dets_ids[ious_inds[attack_ind]])
                    attack_inds.append(attack_ind)
                    target_inds.append(ious_inds[attack_ind])
                elif track_id in self.low_iou_ids and ious[attack_ind, ious_inds[attack_ind]] == 0:
                    self.low_iou_ids.remove(track_id)

            fit_index = self.CheckFit(dets, id_feature, attack_ids, attack_inds)
            if fit_index:
                attack_ids = np.array(attack_ids)[fit_index]
                target_ids = np.array(target_ids)[fit_index]
                attack_inds = np.array(attack_inds)[fit_index]
                target_inds = np.array(target_inds)[fit_index]

                noise_, attack_iter = self.ifgsm_adam_mt(
                    im_blob,
                    img0,
                    id_features,
                    dets,
                    inds,
                    remain_inds,
                    last_info=self.ad_last_info,
                    outputs_ori=output,
                    attack_ids=attack_ids,
                    attack_inds=attack_inds,
                    target_ids=target_ids,
                    target_inds=target_inds
                )
                if noise_ is not None:
                    self.attacked_ids.update(set(attack_ids))
                    self.low_iou_ids.update(set(attack_ids))
                    print(
                        f'attack frame {self.frame_id_}: SUCCESS\tl2 distance: {(noise_ ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                    noise = noise_
                else:
                    print(f'attack frame {self.frame_id_}: FAIL')

        l2_dis = (noise ** 2).sum().sqrt().item()
        im_blob = torch.clip(im_blob + noise, min=0, max=1)

        output_stracks_att = self.update(im_blob, img0, **kwargs)

        noise = self.recoverNoise(noise, img0)
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
            feat = id_features[4]
            lst.append(feat[index])
            feat = sum(lst)
            # feat /= (feat ** 2).sum().sqrt()
            feat = feat.cpu().numpy()
            track.update_features_ad(feat)

    def update(self, im_blob, img0, **kwargs):
        self.frame_id += 1
        self_track_id = kwargs.get('track_id', None)
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
        # for strack in strack_pool:
        # strack.predict()
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
                td_[unconfirmed[itracked].track_id][self.frame_id] = (
                unconfirmed[itracked].smooth_feat, detections[idet].smooth_feat)
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
            track.activate(self.kalman_filter, self.frame_id, track_id=self_track_id)
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
                    f_2.append(td_[fi][i][0] @ td_[fj][i][1])
                f1 = sum(f_1) / len(f_1)
                f2 = sum(f_2) / len(f_2)

                sc = 0
                for i in range(len(f_1)):
                    if f_2[i] > f_1[i]:
                        sc += 1
                print(f'f1:{f1}, f2:{f2}, sc:{sc}, len:{len(f_1)}')
                import pdb;
                pdb.set_trace()

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

    def _nms(self, heat, kernel=3):

        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float
        return keep

    def computer_targets(self, boxes, gt_box):

        an_ws = boxes[:, 2]
        an_hs = boxes[:, 3]
        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]

        gt_ws = gt_box[:, 2]
        gt_hs = gt_box[:, 3]
        gt_ctr_x = gt_box[:, 0]
        gt_ctr_y = gt_box[:, 1]

        targets_dx = (gt_ctr_x - ctr_x) / an_ws
        targets_dy = (gt_ctr_y - ctr_y) / an_hs
        targets_dw = np.log(gt_ws / an_ws)
        targets_dh = np.log(gt_hs / an_hs)

        targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).T

        return targets

    def fgsmV2_1(self, im_blob, id_features, last_ad_id_features, dets, outputs_ori, outputs, ori_det, ad_det,
                 epsilon=0.03):
        ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float),
                         np.ascontiguousarray(dets[:, :4], dtype=np.float))
        ious = self.processIoUs(ious)
        loss = 0
        loss_det = 0
        loss_det1 = 0
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

        loss_det1 -= mse(outputs['hm'], outputs_ori['hm'].data)
        # keep=self._nms(outputs_ori['hm'].data)
        # loss_det-=SigmoidFocalLoss(gamma=2.0,alpha=0.25)(outputs['hm'],keep)

        if len(ad_det) > 0 and len(ori_det) > 0:
            '''Detections'''
            ori_detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                              (tlbrs, f) in zip(ori_det[:, :5], id_features[4].cpu().detach().numpy())]
            ad_detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                             (tlbrs, f) in zip(ad_det[:, :5], id_features[4].cpu().detach().numpy())]
        else:
            ori_detections = []
            ad_detections = []

        ad_dets = []
        ori_dets = []
        for_ad_track = []
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        unconfirmed = self.unconfirmed_ad_iou
        tracked_stracks = self.tracked_stracks_ad_iou
        strack_pool = self.strack_pool_ad_iou
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, ori_detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, ori_detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        ori_detections = [ori_detections[i] for i in u_detection]
        ad_detections = [ad_detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, ori_detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            ad_det = ad_detections[idet]
            ori_det = ori_detections[idet]

            ad_dets.append(ad_det)
            ori_dets.append(ori_det)
            for_ad_track.append(track)
        ori_detections = [ori_detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, ori_detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            ad_det = ad_detections[idet]
            ori_det = ori_detections[idet]

            ad_dets.append(ad_det)
            ori_dets.append(ori_det)
            for_ad_track.append(track)
        if len(ad_dets) >= 2:
            ad_dets = [track.to_xyah() for track in ad_dets]
            ad_dets = np.vstack(ad_dets)
            ad_dets[:, 2] = ad_dets[:, 2] * ad_dets[:, 3]
            ori_dets = [track.to_xyah() for track in ori_dets]
            ori_dets = np.vstack(ori_dets)
            ori_dets[:, 2] = ori_dets[:, 2] * ori_dets[:, 3]
            for_ad_track = [track.to_xyah() for track in for_ad_track]
            for_ad_track = np.vstack(for_ad_track)
            for_ad_track[:, 2] = for_ad_track[:, 2] * for_ad_track[:, 3]
            ious = bbox_ious(np.ascontiguousarray(ori_dets, dtype=np.float),
                             np.ascontiguousarray(ori_dets, dtype=np.float))
            ious = self.processIoUs(ious)
            xs, ys = np.where(ious > 0)
            ori_dx = ori_dets[xs]
            ad_dx = ad_dets[xs]
            for_ad_track_x = for_ad_track[xs]

            ori_dy = ori_dets[ys]
            ad_dy = ad_dets[ys]
            for_ad_track_y = for_ad_track[ys]

            ad_dx_targets = torch.from_numpy(self.computer_targets(ad_dx, for_ad_track_y)).float()
            ori_dy_targets = torch.from_numpy(self.computer_targets(ori_dy, for_ad_track_y)).float()

            ad_dy_targets = torch.from_numpy(self.computer_targets(ad_dy, for_ad_track_x)).float()
            ori_dx_targets = torch.from_numpy(self.computer_targets(ori_dx, for_ad_track_x)).float()

            loss_det = loss_det - smoothL1(ad_dx_targets, ori_dy_targets) - smoothL1(ad_dy_targets, ori_dx_targets)

        for key in ['wh', 'reg']:
            loss_det1 -= smoothL1(outputs[key], outputs_ori[key].data)
        loss += loss_det1
        if isinstance(loss, int):
            return torch.zeros_like(im_blob)
        loss.backward()
        noise = im_blob.grad.sign() * epsilon
        noise = gaussianBlurConv(noise)
        return noise

    def ifgsmV21(self, im_blob, id_features, last_ad_id_features, dets, inds, remain_inds, outputs_ori, epsilon=0.03,
                 alpha=0.003):
        noise = self.fgsmV2(im_blob, id_features, last_ad_id_features, dets, epsilon=alpha)  # 9*[X,512],[X,512],[X,6]
        num = int(epsilon / alpha)
        for i in range(num):
            im_blob_ad = torch.clip(im_blob + noise, min=0, max=1).data
            id_features, outputs, ori_det, ad_det = self.forwardFeature1(im_blob_ad, inds, remain_inds, outputs_ori)
            noise += self.fgsmV2_1(im_blob_ad, id_features, last_ad_id_features, dets, outputs_ori, outputs, ori_det,
                                   ad_det, epsilon=alpha)
        return noise

    def forwardFeature1(self, im_blob, inds, remain_inds, outputs_ori):

        im_blob.requires_grad = True
        self.model.zero_grad()
        output = self.model(im_blob)[-1]
        ad_hm = output['hm']
        ori_hm = outputs_ori['hm'].data
        ad_wh = output['wh']
        ori_wh = outputs_ori['wh'].data
        ad_reg = output['reg']
        ori_reg = outputs_ori['reg'].data
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)
        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(0)
                id_features.append(id_feature_exp)

        ad_det, _ = mot_decode(ori_hm, ad_wh, ad_reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        ori_det, _ = mot_decode(ori_hm, ori_wh, ori_reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        ori_det = self.post_process(ori_det.clone(), self.meta)
        ori_det = self.merge_outputs([ori_det])[1]
        ori_det = ori_det[remain_inds]
        ad_det = self.post_process(ad_det.clone(), self.meta)
        ad_det = self.merge_outputs([ad_det])[1]
        ad_det = ad_det[remain_inds]
        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]
        return id_features, output, ori_det, ad_det

    def update_attack_z(self, im_blob, img0, **kwargs):

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
        self.meta = meta
        ''' Step 1: Network forward, get detections & embeddings'''
        # with torch.no_grad():
        im_blob.requires_grad = True
        self.model.zero_grad()
        output = self.model(im_blob)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg'] if self.opt.reg_offset else None
        dets_raw, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        # [1,k,6],[1,k]
        id_features = []
        for i in range(3):
            for j in range(3):
                id_feature_exp = _tranpose_and_gather_feat_expand(id_feature, inds, bias=(i - 1, j - 1)).squeeze(
                    0)  # [k,512]
                id_features.append(id_feature_exp)

        id_feature = _tranpose_and_gather_feat_expand(id_feature, inds)

        id_feature = id_feature.squeeze(0)  # [k,512]

        dets = self.post_process(dets_raw.clone(), meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres  # X
        dets = dets[remain_inds]  # [X,6]
        id_feature = id_feature[remain_inds]  # [X,512]

        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]  # 9*[X,512]

        id_feature = id_feature.detach().cpu().numpy()

        last_id_features = [None for _ in range(len(dets))]
        last_ad_id_features = [None for _ in range(len(dets))]
        dets_index = [i for i in range(len(dets))]
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
        self.unconfirmed_ad_iou = copy.deepcopy(unconfirmed)
        self.tracked_stracks_ad_iou = copy.deepcopy(tracked_stracks)
        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks_)
        self.strack_pool_ad_iou = copy.deepcopy(strack_pool)
        # Predict the current location with KF
        # for strack in strack_pool:
        # strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
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
        output_stracks_ori = [track for track in self.tracked_stracks_ if track.is_activated]

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
        output_stracks = [track for track in self.tracked_stracks_ if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id_))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        # noise = self.fgsmV3(im_blob, id_features, last_id_features, last_ad_id_features, dets)

        # noise = self.fgsmV2(im_blob, id_features, last_ad_id_features, dets)

        # noise = self.fgsmV1(im_blob, id_features, last_id_features, dets)

        # noise = self.fgsm(im_blob, id_features, dets)

        noise = self.ifgsmV21(im_blob, id_features, last_ad_id_features, dets, inds, remain_inds, outputs_ori=output)

        # noise = self.recoverNoise(noise, img0)
        # noise = self.deRecoverNoise(noise, img0)
        im_blob = torch.clip(im_blob + noise, min=0, max=1)
        # im_blob = (im_blob * 255).int().float() / 255
        l2_dis = (noise ** 2).sum().sqrt().item()

        self.update_ad(im_blob, img0, dets_raw, inds, tracks_ad, **kwargs)

        output_stracks = self.update(im_blob, img0, **kwargs)

        noise = self.recoverNoise(noise, img0)
        # import pdb; pdb.set_trace()
        adImg = np.clip(img0 + noise, a_min=0, a_max=255)

        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        noise = (noise * 255).astype(np.uint8)
        return output_stracks_ori, output_stracks, adImg, noise, l2_dis


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
