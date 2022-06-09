import sys
import json
import numpy as np
import itertools

def get_iou(pred, gt):
    """ Get tIoU of two segments
    """
    start_pred, end_pred = pred
    start, end = gt
    intersection = max(0, min(end, end_pred) - max(start, start_pred))
    union = min(max(end, end_pred) - min(start, start_pred), end - start + end_pred - start_pred)
    iou = float(intersection) / (union + 1e-8)

    return iou

def get_miou(predictions, groundtruths):
    """ Get mean IoU
    """
    ious = []
    missing_num = 0
    all_num = len(groundtruths)
    for idx in groundtruths.keys():
        if idx not in predictions:
            missing_num += 1
            continue
        pred = predictions[idx][0]
        ious.append(get_iou(pred['timestamp'], groundtruths[idx]['timestamp']))

    miou = sum(ious) / all_num
    print('Calculating mIOU: total videos: {}, missing videos: {}'.format(all_num, missing_num))
    return miou

def get_recall_at_k(predictions, groundtruths, iou_threshold=0.5, max_proposal_num=5):
    """ Get R@k for all predictions
    R@k: Given k proposals, if there is at least one proposal has higher tIoU than iou_threshold, R@k=1; otherwise R@k=0
    The predictions should have been sorted by confidence
    """
    hit = np.zeros(shape=(len(groundtruths.keys()),), dtype=np.float32)
    all_num = len(groundtruths)
    missing_num = 0
    for idd, idx in enumerate(groundtruths.keys()):
        if idx not in predictions.keys():
            missing_num += 1
        if idx in predictions.keys():
            preds = predictions[idx][:max_proposal_num]
            for pred in preds:
                if get_iou(pred['timestamp'], groundtruths[idx]['timestamp']) >= iou_threshold:
                    hit[idd] = 1.

    avg_recall = np.sum(hit) / len(hit)
    print('Calculating Recall@{}: total videos: {}, missing videos: {}'.format(max_proposal_num, all_num, missing_num))
    return avg_recall



def eval_result(result_file, gt_file, need_eval_multi_anno=False):
    """
    Calculate mIoU, recalls for a given result file
    :param result_file: input .json result file
    :param gt_file: ground-truth file
    :return: None
    """
    results = json.load(open(result_file, 'r'))['results']
    groundtruth_data = json.load(open(gt_file, 'r'))
    if need_eval_multi_anno:
        results, groundtruth_data = process_for_mutli_anno(results, groundtruth_data)
    else:
        video_ids = list(groundtruth_data.keys())
        out_grounding_data = {}
        for video_id in video_ids:
            gd = groundtruth_data[video_id]
            for anno_id in range(len(gd['timestamps'])):
                unique_anno_id = video_id + '-' + str(anno_id)
                out_grounding_data[unique_anno_id] = {
                    'video_id': video_id,
                    'anno_id': anno_id,
                    'timestamp': gd['timestamps'][anno_id]
                }
        groundtruth_data = out_grounding_data

    miou = get_miou(results, groundtruth_data)
    print('mIoU: {}'.format(miou))
    scores = {}
    scores['mIOU'] = miou

    for iou, max_proposal_num in list(itertools.product([0.7, 0.5, 0.3, 0.1], [1, 5])):
        recall = get_recall_at_k(results, groundtruth_data, iou_threshold=iou, max_proposal_num=max_proposal_num)
        print('R@{}, IoU={}: {}'.format(max_proposal_num, iou, recall))
        scores['R@{}IOU{}'.format(max_proposal_num, iou)] = recall
    return scores


def process_for_mutli_anno(prediction, groundtruth):
    '''
    :param prediction: {rebuild_video_key-anno_id: [{timestep: [s, e], score: confidence, sentence: Input sentence}]}
    :param groundtruth: {rebuild_video_key: {timestep: [[s1, e1], ..., [sn, en]], sentences: [S1, ... Sn]}}
    '''
    reformated_gt_data = {}
    # reformat grou 
    for video_key, video_gt_info in groundtruth.items():
        for anno_id in range(len(video_gt_info['timestamps'])):
            unique_anno_id = video_key + '-' + str(anno_id)
            reformated_gt_data[unique_anno_id] = {
                'origin_video_key': video_key[3:],
                'timestamp': video_gt_info['timestamps'][anno_id],
                'sentence': video_gt_info['sentences'][anno_id]
            }
    # {origin_video_key: {sentence: {gt, pred, score}}}
    temp_gathered_data = {}
    for video_key in reformated_gt_data.keys():
        gt = reformated_gt_data[video_key]
        pred = prediction[video_key][0]
        assert gt['sentence'] == pred['sentence']
        origin_video_key = gt['origin_video_key']
        if origin_video_key not in temp_gathered_data:
            temp_gathered_data[origin_video_key] = {}
        gather_item = {
            'gt_timestamp': gt['timestamp'],
            'pred_timestamp': pred['timestamp'],
            'pred_score': pred['score']
        }
        if pred['sentence'] in temp_gathered_data[origin_video_key]:
            if gather_item['pred_score'] > temp_gathered_data[origin_video_key][pred['sentence']]['pred_score']:
                temp_gathered_data[origin_video_key][pred['sentence']] = gather_item
        else:
            temp_gathered_data[origin_video_key][pred['sentence']] = gather_item
            

    gathered_pred = {}
    gathered_gt = {}
    for video_key, video_info in temp_gathered_data.items():
        for anno_id, sent in enumerate(video_info.keys()):
            anno_info = video_info[sent]
            unique_anno_id = video_key + '-' + str(anno_id)
            gathered_pred[unique_anno_id] = [{'timestamp':anno_info['pred_timestamp']}]
            gathered_gt[unique_anno_id] = {'timestamp':anno_info['gt_timestamp']}
    return gathered_pred, gathered_gt



if __name__ == '__main__':
    eval_result(sys.argv[1], sys.argv[2], sys.argv[3])
