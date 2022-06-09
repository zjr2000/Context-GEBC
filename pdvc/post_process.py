import json
import numpy as np
p='/Users/wangteng/PycharmProjects/jirui_project/VideoDETR_jinrui/save/0305_M100_RL/2022-05-06-14-41-35_05_06debug_2022-05-06_14-33-09_epoch6_num4917_alpha1.0.json'
gt_p= '/Users/wangteng/PycharmProjects/anet_challenge_2022/data/anet/captiondata/val_1.json'
d=json.load(open(p))
gt_d = json.load(open(gt_p))
