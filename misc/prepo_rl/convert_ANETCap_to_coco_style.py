import json


def tokenize(sentence):
    tokens = [',', ':', '!', '_', ';', '.', '?', '"', '\\n', '\\', '.']
    for token in tokens:
        sentence = sentence.replace(token, ' ')
    sentence_split = sentence.replace('.', ' . ').replace(',', ' , ').lower().split()
    return sentence_split

def para_level():
    p_train='data/gebc/train_all_annotation_subject.json'
    p_val1 = 'data/gebc/valset_highest_f1_subject.json'
    # p_val2 = 'data/captiondata/val_2.json'
    p_out = 'data/gebc/rl_files/dataset_gebc_subject_cocostyle.json'

    d_train = json.load(open(p_train))
    d_val1 = json.load(open(p_val1))
    # d_val2 = json.load(open(p_val2))

    out = {"images":{}}
    images = []
    img_id = 0
    sent_id = 0

    for d,split in [(d_train, 'train'), (d_val1, 'val')]:
        for vid, v in d.items():
            v_imgid = img_id
            img_id +=1
            sentences = []
            for s in v['sentences']:
                prop_sent_id = sent_id
                sent_id+=1

                raw_sent = s
                tokens = tokenize(s)

                sentences.append({'tokens': tokens,
                                  'raw': raw_sent,
                                  'imgid': v_imgid,
                                  'sentid': prop_sent_id})
            vid_info = {'imgid': v_imgid,
                        'filename': vid,
                        'split': split,
                        'sentences': sentences}
            images.append(vid_info)

    out = {'images': images,
           'dataset': 'activity_caption',}
    json.dump(out, open(p_out, 'w'))

def sentence_level():
    # p_train='data/captiondata/train_modified.json'
    # p_val = 'data/captiondata/val_all.json'
    p_train='data/gebc/train_all_annotation_after.json'
    p_train2='data/gebc/train_all_annotation_before.json'
    p_val = 'data/gebc/valset_highest_f1_after.json'
    p_val2 = 'data/gebc/valset_highest_f1_before.json'
    p_out = 'data/gebc/rl_files/dataset_gebc_before+after_cap_cocostyle.json'

    d_train = json.load(open(p_train))
    d_train2 = json.load(open(p_train2))
    d_val = json.load(open(p_val))
    d_val2 = json.load(open(p_val2))

    out = {"images":{}}
    images = []
    img_id = 0
    sent_id = 0

    for d,split in [(d_train, 'train'), (d_val, 'val'), (d_train2, 'train'), (d_val2, 'val')]:
        for vid, v in d.items():
            for i,s in enumerate(v['sentences']):
                v_imgid = img_id
                img_id += 1
                sentences = []
                prop_sent_id = sent_id
                sent_id+=1

                raw_sent = s
                tokens = tokenize(s)

                sentences.append({'tokens': tokens,
                                  'raw': raw_sent,
                                  'imgid': v_imgid,
                                  'sentid': prop_sent_id})
                vid_info = {'imgid': v_imgid,
                            'filename': vid + '_' + str(i),
                            'split': split,
                            'sentences': sentences}
                images.append(vid_info)

    out = {'images': images,
           'dataset': 'activity_caption',}
    json.dump(out, open(p_out, 'w'))


if __name__=='__main__':
    sentence_level()