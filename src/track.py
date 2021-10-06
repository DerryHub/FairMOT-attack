# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

from cython_bbox import bbox_overlaps as bbox_ious
from scipy.optimize import linear_sum_assignment

class TrackObject:
    def __init__(self, result_lines, id):
        self.dic = {}
        self.id = id
        self.frames = []
        for line in result_lines:
            line = list(map(float, line.strip().split(',')))
            if int(line[1]) != id:
                continue
            assert int(line[0]) not in self.dic
            self.frames.append(int(line[0]))
            self.dic[int(line[0])] = {
                'xywh': np.array(line[2:6]),
                'match': None
            }

    def getXYWH(self, frame_id):
        if frame_id not in self.dic:
            return None
        return self.dic[frame_id]['xywh']

    def updateMatch(self, frame_id, track):
        assert frame_id in self.dic and self.dic[frame_id]['match'] is None
        self.dic[frame_id]['match'] = track

    @property
    def length(self):
        return len(self.dic)

    def __repr__(self):
        s = ''
        for frame_id in self.frames:
            if self.dic[frame_id]['match'] is None:
                s += f"frame_id: {frame_id}, xywh: {self.dic[frame_id]['xywh']}, match_id: -1\n"
            else:
                s += f"frame_id: {frame_id}, xywh: {self.dic[frame_id]['xywh']}, " \
                     f"match_id: {self.dic[frame_id]['match'].id}\n"
        return s



def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def show(img, dets):
    for det in dets:
        det = det[0]
        img = cv2.rectangle(img, (int(det[0]), int(det[1])), (int(det[0]+det[2]), int(det[1]+det[3])), color=(255,1,1))
    return img


def decodeResult(result_filename):
    with open(result_filename, 'r') as f:
        lines = f.readlines()
    ids = set([])
    for line in lines:
        line = list(map(float, line.strip().split(',')))
        ids.add(int(line[1]))
    tracks = []
    for id in ids:
        tracks.append(TrackObject(lines, id))
    return tracks, int(line[0]), sorted(list(ids))


def decodeTrack(tracks, frame):
    ids = []
    xywhs = np.zeros([0, 4])
    tracks_frame = []
    for track in tracks:
        if track.getXYWH(frame) is None:
            continue
        xywhs = np.append(xywhs, track.getXYWH(frame).reshape(1, -1), axis=0)
        ids.append(track.id)
        tracks_frame.append(track)
    tlbrs = xywhs.copy()
    tlbrs[:, 2:] += tlbrs[:, :2]
    return ids, tlbrs, tracks_frame


def evaluate_attack(result_filename_ori, result_filename_att):
    ori_tracks, frames_o, ori_all_ids = decodeResult(result_filename_ori)
    att_tracks, frames_a, att_all_ids = decodeResult(result_filename_att)
    assert frames_a == frames_o
    frames = frames_o
    track_union = np.zeros([len(ori_all_ids), len(att_all_ids)])
    ori_track_len = np.zeros(len(ori_all_ids))
    att_track_len = np.zeros(len(att_all_ids))
    for track in ori_tracks:
        ori_track_len[ori_all_ids.index(track.id)] = track.length
    for track in att_tracks:
        att_track_len[att_all_ids.index(track.id)] = track.length
    for frame in range(1, frames + 1):
        ori_ids, ori_tlbrs, ori_tracks_frame = decodeTrack(ori_tracks, frame)
        att_ids, att_tlbrs, att_tracks_frame = decodeTrack(att_tracks, frame)
        ious = -bbox_ious(ori_tlbrs, att_tlbrs)
        row_inds, col_inds = linear_sum_assignment(ious)
        for row_ind, col_ind in zip(row_inds, col_inds):
            if ious[row_ind, col_ind] == 0:
                continue
            ori_tracks_frame[row_ind].updateMatch(frame, att_tracks_frame[col_ind])
            att_tracks_frame[col_ind].updateMatch(frame, ori_tracks_frame[row_ind])
            track_union[ori_all_ids.index(ori_ids[row_ind]), att_all_ids.index(att_ids[col_ind])] += 1
    ori_track_len = ori_track_len.reshape([-1, 1]).repeat(len(att_all_ids), axis=1)
    att_track_len = att_track_len.reshape([1, -1]).repeat(len(ori_all_ids), axis=0)
    track_iou = track_union / (ori_track_len + att_track_len - track_union)
    mean_recall = track_union.sum() / ori_track_len[:, 0].sum()
    mean_precision = track_union.sum() / att_track_len[0].sum()
    mean_iou = track_iou.max(axis=1).mean()
    return mean_recall, mean_precision, mean_iou


def eval_seq(opt, dataloader, data_type, result_filename, gt_dict, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    results_att = []
    frame_id = 0
    root_r = opt.data_dir
    root_r += '/' if root_r[-1] != '/' else ''
    root = opt.output_dir
    root += '/' if root[-1] != '/' else ''
    imgRoot = os.path.join(root, 'image')
    noiseRoot = os.path.join(root, 'noise')
    l2_distance = []
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        if opt.attack:
            if opt.attack == 'single':
                online_targets_ori, output_stracks_att, adImg, noise, l2_dis = tracker.update_attack_sg(blob, img0,
                                                                                                     name=path.replace(
                                                                                                         root_r, ''))
            elif opt.attack == 'multiple':
                online_targets_ori, output_stracks_att, adImg, noise, l2_dis = tracker.update_attack_mt(blob, img0,
                                                                                                     name=path.replace(
                                                                                                         root_r, ''))
            else:
                raise RuntimeError()
            imgPath = os.path.join(imgRoot, path.replace(root_r, ''))
            os.makedirs(os.path.split(imgPath)[0], exist_ok=True)
            noisePath = os.path.join(noiseRoot, path.replace(root_r, ''))
            os.makedirs(os.path.split(noisePath)[0], exist_ok=True)

            l2_distance.append(l2_dis)

            cv2.imwrite(imgPath, adImg)
            cv2.imwrite(noisePath, noise)

            online_tlwhs_att = []
            online_ids_att = []
            for t in output_stracks_att:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs_att.append(tlwh)
                    online_ids_att.append(tid)
            results_att.append((frame_id + 1, online_tlwhs_att, online_ids_att))
        else:
            online_targets_ori = tracker.update(blob, img0, name=path.replace(root_r, ''))

        online_tlwhs = []
        online_ids = []
        for t in online_targets_ori:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            if opt.attack:
                img0 = adImg.astype(np.uint8)
                online_im = vis.plot_tracking(img0, online_tlwhs_att, online_ids_att, frame_id=frame_id,
                                              fps=1. / timer.average_time)
            else:
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                              fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            save_dir = os.path.join(imgRoot, save_dir.replace(root_r, ''))
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    if opt.attack:
        write_results(result_filename.replace('.txt', '_attack.txt'), results_att, data_type)
    return frame_id, timer.average_time, timer.calls, l2_distance


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    root_r = opt.data_dir
    root_r += '/' if root_r[-1] != '/' else ''
    root = opt.output_dir
    root += '/' if root[-1] != '/' else ''
    result_root = os.path.join(opt.output_dir, 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    accs_att = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        evaluator = Evaluator(data_root, seq, data_type)
        gt_frame_dict = evaluator.gt_frame_dict
        gt_ignore_frame_dict = evaluator.gt_ignore_frame_dict

        # evaluate_attack(result_filename, result_filename.replace('.txt', '_attack.txt'))
        # import pdb; pdb.set_trace()

        nf, ta, tc, l2_distance = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate, gt_dict=gt_frame_dict)


        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))

        if opt.attack:
            mean_recall, mean_precision, mean_iou = evaluate_attack(result_filename,
                                                                    result_filename.replace('.txt', '_attack.txt'))
            logger.info(f'mean_recall: {mean_recall}\tmean_precision: {mean_precision}\tmean_iou: {mean_iou}\t'
                        f'mean_l2_distance: {sum(l2_distance) / len(l2_distance)}\n')

        accs.append(evaluator.eval_file(result_filename))
        if opt.attack:
            accs_att.append(evaluator.eval_file(result_filename.replace('.txt', '_attack.txt')))
        if save_videos:
            if opt.attack:
                output_dir = output_dir.replace(root_r, os.path.join(root, 'image/'))
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    # import pdb; pdb.set_trace()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print('=' * 50 + 'origin' + '=' * 50)
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))
    if opt.attack:
        summary_att = Evaluator.get_summary(accs_att, seqs, metrics)
        strsummary_att = mm.io.render_summary(
            summary_att,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print('=' * 50 + 'attack' + '=' * 50)
        print(strsummary_att)
        Evaluator.save_summary(summary_att, os.path.join(result_root, 'summary_{}_attack.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if opt.attack == 'single':
        opt.output_dir = os.path.join(opt.output_dir, f'{opt.attack}_{opt.attack_id}')
    elif opt.attack == 'multiple':
        opt.output_dir = os.path.join(opt.output_dir, opt.attack)
    elif not opt.attack:
        opt.output_dir = os.path.join(opt.output_dir, 'origin')
    else:
        raise RuntimeError()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        # seqs_str = '''MOT17-02-SDP
        #               MOT17-04-SDP
        #               MOT17-05-SDP
        #               MOT17-09-SDP
        #               MOT17-10-SDP
        #               MOT17-11-SDP
        #               MOT17-13-SDP'''
        seqs_str = '''MOT17-04-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        # seqs_str = '''KITTI-13
        #               KITTI-17
        #               ETH-Bahnhof
        #               ETH-Sunnyday
        #               PETS09-S2L1
        #               TUD-Campus
        #               TUD-Stadtmitte
        #               ADL-Rundle-6
        #               ADL-Rundle-8
        #               ETH-Pedcross2
        #               TUD-Stadtmitte'''
        seqs_str = '''PETS09-S2L1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='all_dla34',
         show_image=False,
         save_images=True,
         save_videos=True if opt.attack else True)
