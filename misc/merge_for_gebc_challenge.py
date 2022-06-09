import pdb
import json
import pickle

if False:
    p_subject = 'save/05_29_gebc_clip_sub_lr5e5_FullTrain_v_2022-05-29-05-06-48_/2022-05-29-13-48-54_05_29_gebc_clip_sub_lr5e5_FullTrain_v_2022-05-29-05-06-48__epoch5_num2082_alpha1.0.json'
    p_before = 'save/05_29_gebc_clip_before_lr5e5_FullTrain_v_2022-05-29-06-05-57_/2022-05-29-14-44-12_05_29_gebc_clip_before_lr5e5_FullTrain_v_2022-05-29-06-05-57__epoch19_num2082_alpha1.0.json'
    p_after = 'save/05_29_gebc_clip_after_lr5e5_FullTrain_v_2022-05-29-06-06-04_/2022-05-29-14-47-44_05_29_gebc_clip_after_lr5e5_FullTrain_v_2022-05-29-06-06-04__epoch24_num2082_alpha1.0.json'
    p_output = 'save/000_gebc_submission/submit_test.pkl'
    out = {}

if True:
    p_subject = 'save/05_29_gebc_clip_sub_lr5e5_FullTrain_v_2022-05-29-05-06-48_/prediction/num2081_epoch5.json'
    p_before = 'save/05_29_gebc_clip_before_lr5e5_FullTrain_v_2022-05-29-06-05-57_/prediction/num2081_epoch19.json'
    p_after = 'save/05_29_gebc_clip_after_lr5e5_FullTrain_v_2022-05-29-06-06-04_/prediction/num2081_epoch24.json'
    p_output = 'save/000_gebc_submission/submit_val.pkl'
    out = {'oc2bib7RLLo_0': {'Subject': '', 'Status_Before': '', 'Status_After': ''},
            'oc2bib7RLLo_1': {'Subject': '', 'Status_Before': '', 'Status_After': ''},
            'oc2bib7RLLo_2': {'Subject': '', 'Status_Before': '', 'Status_After': ''}}


keys = ['Subject', 'Status_Before', 'Status_After']
paths = [p_subject, p_before, p_after]

captions = [json.load(open(p))['results'] for p in paths]
all_vids = captions[0].keys()
for vid in all_vids:
    for key, caption in zip(*[keys, captions]):
        caption_vid = caption[vid]
        caption_vid = sorted(caption_vid, key=lambda x: x['query_id'])
        for b_id, b_sent in enumerate(caption_vid):
            b_vid = vid[2:] + '_{}'.format(b_id)
            if b_vid not in out:
                out[b_vid] = {}
            out[b_vid][key] = b_sent['sentence'].rstrip('.')

pickle.dump(out, open(p_output, 'wb'))