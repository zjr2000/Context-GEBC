# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
import time
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import sys
import collections
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import dirname, abspath
from pdvc.pdvc import build 

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
# print(sys.path)


from eval_utils import evaluate
import opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from video_dataset import PropSeqDataset, collate_fn
from pdvc.pdvc import build
from collections import OrderedDict
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from video_dataset import PropSeqDataset as PropSeqDataset_train
from video_dataset import collate_fn as collate_fn_train

def build_scheduler(opt, optimizer, total_steps):
    if opt.learning_strategy == 'warmup_linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opt.warm_up_ratio*total_steps,
            num_training_steps=total_steps
        )
    elif opt.learning_strategy == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opt.warm_up_ratio*total_steps,
            num_training_steps=total_steps
        )
    elif opt.learning_strategy == 'multi_step':
        milestone = [opt.learning_rate_decay_start + opt.learning_rate_decay_every * _ for _ in
                     range(int((opt.epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every))]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=opt.learning_rate_decay_rate)
    else:
        raise NotImplementedError()
    return scheduler

def build_text_encoder_scheduler(opt, optimizer, total_steps):
    if opt.text_encoder_learning_strategy == 'warmup_linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opt.text_encoder_warm_up_ratio*total_steps, 
            num_training_steps=total_steps
        )
    elif opt.text_encoder_learning_strategy == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opt.text_encoder_warm_up_ratio*total_steps, 
            num_training_steps=total_steps
        )
    elif opt.text_encoder_learning_strategy == 'multi_step':
        milestone = [opt.text_encoder_lr_decay_start + opt.text_encoder_lr_decay_every * _ for _ in range(int((opt.epoch - opt.text_encoder_lr_decay_start) / opt.text_encoder_lr_decay_every))]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=opt.text_encoder_lr_decay_rate)
    else:
        raise AssertionError('Undefined text encoder scheduler type')
    return scheduler


def update_task_best_score_details(task_name, task_details, eval_score):
    if task_name == 'dvc':
        task_details['CIDEr'] = np.array(eval_score['CIDEr']).mean()
        task_details['REOUGE_L'] = np.array(eval_score['ROUGE_L']).mean()
        task_details['SPICE'] = np.array(eval_score['SPICE']).mean()
    elif task_name == 'pc':
        task_details['para_METEOR'] = np.array(eval_score['para_METEOR']).mean()
        task_details['para_CIDEr'] = np.array(eval_score['para_CIDEr']).mean()
        task_details['para_Bleu_4'] = np.array(eval_score['para_Bleu_4']).mean()
    else:
        raise AssertionError('Undefined task')


def train(opt):
    set_seed(opt.seed)
    save_folder = build_floder(opt)
    logger = create_logger(save_folder, 'train.log')
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))

    if not opt.start_from:
        backup_envir(save_folder)
        logger.info('backup evironment completed !')

    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    # continue training
    if opt.start_from:
        opt.pretrain = False
        infos_path = os.path.join(save_folder, 'info.json')
        with open(infos_path) as f:
            logger.info('Load info from {}'.format(infos_path))
            saved_info = json.load(f)
            prev_opt = saved_info[opt.start_from_mode[:4]]['opt']

            exclude_opt = ['start_from', 'start_from_mode', 'pretrain', 'debug', 'nthreads']
            for opt_name in prev_opt.keys():
                if opt_name not in exclude_opt:
                    vars(opt).update({opt_name: prev_opt.get(opt_name)})
                if prev_opt.get(opt_name) != vars(opt).get(opt_name):
                    logger.info('Change opt {} : {} --> {}'.format(opt_name, prev_opt.get(opt_name),
                                                                   vars(opt).get(opt_name)))

    train_dataset = PropSeqDataset_train(opt.train_caption_file,
                                   opt.visual_feature_folder,
                                   opt.dict_file, True, 'gt',
                                   opt)

    val_dataset = PropSeqDataset(opt.val_caption_file,
                                 opt.visual_feature_folder,
                                 opt.dict_file, False, 'gt',
                                 opt)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn_train, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=opt.eval_batch_size,
                            shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)

    epoch = saved_info[opt.start_from_mode[:4]].get('epoch', 0)
    iteration = saved_info[opt.start_from_mode[:4]].get('iter', 0)
    best_val_score = saved_info[opt.start_from_mode[:4]].get('best_val_score', -1e5)
    best_dvc_score = saved_info[opt.start_from_mode[:4]].get('best_dvc_score', -1e5)
    best_pc_score = saved_info[opt.start_from_mode[:4]].get('best_pc_score', -1e5)
    best_localization_score = saved_info[opt.start_from_mode[:4]].get('best_localization_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    opt.current_lr = vars(opt).get('current_lr', opt.lr)

    print_opt(opt, None, logger)

    best_dvc_details = {}
    best_pc_details = {}

    # Build model
    model, criterion, constrastive_criterion, postprocessors = build(opt)
    model.translator = train_dataset.translator
    model.train()

    # Recover the parameters
    if opt.start_from and (not opt.pretrain):
        if opt.start_from_mode == 'best':
            model_pth = torch.load(os.path.join(save_folder, 'model-best.pth'))
        elif opt.start_from_mode == 'last':
            model_pth = torch.load(os.path.join(save_folder, 'model-last.pth'))
        logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
        model.load_state_dict(model_pth['model'])

    # Load the pre-trained model
    if opt.pretrain and (not opt.start_from):
        logger.info('Load pre-trained parameters from {}'.format(opt.pretrain_path))
        model_pth = torch.load(opt.pretrain_path, map_location=torch.device(opt.device))
        # query_weight = model_pth['model'].pop('query_embed.weight')
        if opt.pretrain == 'encoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            encoder_pth = {k:v for k,v in model_pth['model'].items() if encoder_filter(k)}
            model.load_state_dict(encoder_pth, strict=True)
        elif opt.pretrain == 'decoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            decoder_pth = {k:v for k,v in model_pth['model'].items() if not encoder_filter(k)}
            model.load_state_dict(decoder_pth, strict=True)
            pass
        elif opt.pretrain == 'full':
            # model_pth = transfer(model, model_pth)
            if opt.only_ft_class_head:
                checkpoint_model = model_pth['model']
                state_dict = model.state_dict()
                for k in ['class_head.0.weight', 'class_head.0.bias',
                'class_head.1.weight', 'class_head.1.bias']:
                    # if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    logger.info("Removing key {} from pretrained checkpoint".format(k))
                    del checkpoint_model[k]
                model.load_state_dict(checkpoint_model, strict=False)

            elif opt.ft_captioner_from_scratch:
                checkpoint_model = model_pth['model']
                delete = []
                for key in checkpoint_model.keys():
                    if key.startswith('caption_head'):
                        delete.append(key)
                for key in delete:
                    logger.info("Removing key {} from pretrained checkpoint".format(key))
                    del checkpoint_model[key]
                model.load_state_dict(checkpoint_model, strict=False)
            elif opt.only_ft_selector_head:
                model.load_state_dict(model_pth['model'], strict=False)
            else:
                model.load_state_dict(model_pth['model'], strict=True)
        else:
            raise ValueError("wrong value of opt.pretrain")

    model.to(opt.device)
    if opt.enable_contrastive: 
        text_encoder_params = list(map(id, model.text_encoder.parameters()))
        other_params = filter(lambda p: id(p) not in text_encoder_params, model.parameters())
    else:
        other_params = model.parameters()

    if opt.only_ft_captioner:
        other_params = model.captioner_parameters()

    if opt.ft_captioner_from_scratch:
        for _, p in model.named_parameters():
            p.requires_grad = False
        # only tune class_head parameters
        for _, p in model.caption_head.named_parameters():
            p.requires_grad = True
        other_params = model.captioner_parameters()

    if opt.only_ft_class_head:
        # Frozen feature level parameters
        for _, p in model.named_parameters():
            p.requires_grad = False
        # only tune class_head parameters
        for _, p in model.class_head.named_parameters():
            p.requires_grad = True
        other_params = model.class_head_paramenters()

    if opt.only_ft_selector_head:
        # Frozen feature level parameters
        for _, p in model.named_parameters():
            p.requires_grad = False
        # only tune class_head parameters
        for p in model.selector_parameters():
            p.requires_grad = True
        other_params = model.selector_parameters()
    
    if opt.enable_contrastive and opt.text_encoder_learning_strategy == 'frozen':
        for _, p in model.text_encoder.named_parameters():
            p.requires_grad = False

    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(params=other_params, lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(params=other_params, lr=opt.lr, weight_decay=opt.weight_decay)

    need_update_text_encoder = opt.enable_contrastive and opt.text_encoder_learning_strategy != 'frozen'
    if need_update_text_encoder:
        if opt.optimizer_type == 'adam':
            text_encoder_optimizer = optim.Adam(params=model.text_encoder.parameters(), lr=opt.text_encoder_lr, weight_decay=opt.weight_decay)
        elif opt.optimizer_type == 'adamw':
            text_encoder_optimizer = optim.AdamW(params=model.text_encoder.parameters(), lr=opt.text_encoder_lr, weight_decay=opt.weight_decay)
        total_steps = int(opt.epoch * len(train_loader))
        text_encoder_scheduler = build_text_encoder_scheduler(opt, text_encoder_optimizer, total_steps)
    total_steps = int(opt.epoch * len(train_loader))
    lr_scheduler = build_scheduler(opt, optimizer, total_steps)
    cl_schedule_time = opt.cl_schedule_time
    cl_schedule_val = opt.cl_schedule_val
    cl_weight = 0.0
    # if start_from recover current cl weight
    for i in range(1, len(cl_schedule_val)):
        if epoch >= cl_schedule_time[i-1] and epoch < cl_schedule_time[i]:
            cl_weight = cl_schedule_val[i-1]
            break 

    # Load tokenizer for text encoder
    if opt.enable_contrastive:
        for i in range(10):
            try:
                tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_language_model, cache_dir=opt.huggingface_cache_dir)
                break
            except:
                print('download error in AutoTokenizer, retry...')
                time.sleep(1)
    else:
        tokenizer = None

    if opt.start_from:
        optimizer.load_state_dict(model_pth['optimizer'])
        if opt.learning_strategy == 'multi_step':
            lr_scheduler.step(epoch-1)
        else:
            lr_scheduler.step((epoch-1)*len(train_dataset))
        if need_update_text_encoder:
            text_encoder_optimizer.load_state_dict(model_pth['text_encoder_optimizer'])
            if opt.text_encoder_learning_strategy == 'multi_step':
                text_encoder_scheduler.step(epoch-1)
            else:
                text_encoder_scheduler.step((epoch-1)*len(train_dataset))

    # print the args for debugging
    print_opt(opt, model, logger)
    print_alert_message('Strat training !', logger)

    loss_sum = OrderedDict()
    bad_video_num = 0

    start = time.time()
    for key, val in criterion.weight_dict.items():
        if 'contrastive_loss' in key:
            criterion.weight_dict[key] = cl_weight
            criterion.matcher.cost_cl = 0 if cl_weight == 0 else opt.set_cost_cl

    weight_dict = criterion.weight_dict
    logger.info('loss type: {}'.format(weight_dict.keys()))
    logger.info('loss weights: {}'.format(weight_dict.values()))

    # Epoch-level iteration
    while True:
        if True:
            # scheduled sampling rate update
            if epoch > opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac,
                                  opt.scheduled_sampling_max_prob)
                model.caption_head.ss_prob = opt.ss_prob

            print('lr:{}'.format(float(opt.current_lr)))
            pass

        if epoch in cl_schedule_time:
            cl_weight = cl_schedule_val[cl_schedule_time.index(epoch)]
            for key, val in weight_dict.items():
                if 'contrastive_loss' in key:
                    weight_dict[key] = cl_weight
                    criterion.matcher.cost_cl = 0 if cl_weight == 0 else opt.set_cost_cl
            logger.info('Update loss weight !')
            logger.info('Loss type: {}'.format(weight_dict.keys()))
            logger.info('Loss weights: {}'.format(weight_dict.values()))


        # Batch-level iteration
        for dt in tqdm(train_loader, disable=opt.disable_tqdm):
            if opt.device=='cuda':
                torch.cuda.synchronize(opt.device)
            if opt.debug:
                # each epoch contains less mini-batches for debugging
                if (iteration + 1) % 5 == 0:
                    iteration += 1
                    break
            iteration += 1

            optimizer.zero_grad()
            if need_update_text_encoder: 
                text_encoder_optimizer.zero_grad()
            dt = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt['video_target'] = [
                {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                dt['video_target']]

            if opt.enable_contrastive:
                captions = list()
                for video_sents in dt['cap_raw']:
                    captions.extend(video_sents)
                text_encoder_input = tokenizer(captions, return_tensors='pt', truncation=True, padding=True, max_length=opt.max_text_input_len)
                text_encoder_input = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in text_encoder_input.items()}
                dt['text_encoder_input'] = text_encoder_input

            dt = collections.defaultdict(lambda: None, dt)

            output, loss = model(dt, criterion, constrastive_criterion, opt.transformer_input_type)

            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()
            if opt.learning_strategy != 'multi_step':
                lr_scheduler.step()
            if need_update_text_encoder: 
                text_encoder_optimizer.step()
                if opt.text_encoder_learning_strategy != 'multi_step':
                    text_encoder_scheduler.step()


            for loss_k,loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0)+ loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()

            if opt.device=='cuda':
                torch.cuda.synchronize()

            losses_log_every = int(len(train_loader) / 10)

            if opt.debug:
                losses_log_every = 6

            if iteration % losses_log_every == 0:
                end = time.time()
                for k in loss_sum.keys():
                    loss_sum[k] = np.round(loss_sum[k] /losses_log_every, 3).item()

                logger.info(
                    "ID {} iter {} (epoch {}), \nloss = {}, \ntime/iter = {:.3f}, bad_vid = {:.3f}"
                        .format(opt.id, iteration, epoch, loss_sum,
                                (end - start) / losses_log_every, bad_video_num))
                if need_update_text_encoder:
                    text_encoder_lr = text_encoder_optimizer.param_groups[0]['lr']
                    tf_writer.add_scalar('text_encoder_lr', text_encoder_lr, iteration)
                opt.current_lr = optimizer.param_groups[0]['lr']
                tf_writer.add_scalar('lr', opt.current_lr, iteration)
                for loss_type in loss_sum.keys():
                    tf_writer.add_scalar(loss_type, loss_sum[loss_type], iteration)
                loss_history[iteration] = loss_sum
                lr_history[iteration] = opt.current_lr
                loss_sum = OrderedDict()
                start = time.time()
                bad_video_num = 0
                torch.cuda.empty_cache()

        # evaluation
        if (epoch % opt.save_checkpoint_every == 0) and (epoch >= opt.min_epoch_when_save):

            # Save model
            saved_pth = {'epoch': epoch,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(), }
            if need_update_text_encoder:
                saved_pth['text_encoder_optimizer'] = text_encoder_optimizer.state_dict()

            if opt.save_all_checkpoint:
                checkpoint_path = os.path.join(save_folder, 'model_iter_{}.pth'.format(iteration))
            else:
                checkpoint_path = os.path.join(save_folder, 'model-last.pth')

            torch.save(saved_pth, checkpoint_path)

            model.eval()
            result_json_path = os.path.join(save_folder, 'prediction',
                                         'num{}_epoch{}.json'.format(
                                             len(val_dataset), epoch))
            # if opt.enable_selector:
                # _, _ = evaluate(model, criterion, constrastive_criterion ,postprocessors, val_loader, result_json_path, logger=logger, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug, tokenizer=tokenizer, disable_selector=True)
            eval_score, eval_loss = evaluate(model, criterion, constrastive_criterion ,postprocessors, val_loader, result_json_path, logger=logger, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug, tokenizer=tokenizer, dvc_eval_version=opt.eval_tool_version)

            current_localization_score = 2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
            current_dvc_score = np.array(eval_score['CIDEr']).mean() + np.array(eval_score['ROUGE_L']).mean() + np.array(eval_score['SPICE']).mean()
            current_pc_score = np.array(eval_score['para_METEOR']).mean() + np.array(eval_score['para_CIDEr']).mean() + np.array(eval_score['para_Bleu_4']).mean()
            if opt.caption_decoder_type == 'none':
                current_score = current_localization_score
            else:
                current_score = current_dvc_score if opt.criteria_for_best_ckpt == 'dvc' else current_pc_score

            best_suffix_list = []
            if current_dvc_score > best_dvc_score:
                update_task_best_score_details('dvc', best_dvc_details, eval_score)
                best_dvc_score = current_dvc_score
                best_suffix_list.append('dvc')
            if current_pc_score > best_pc_score: 
                update_task_best_score_details('pc', best_pc_details, eval_score)
                best_pc_score = current_pc_score
                best_suffix_list.append('pc')
            # add to tf summary
            for key in eval_score.keys():
                tf_writer.add_scalar(key, np.array(eval_score[key]).mean(), iteration)

            for loss_type in eval_loss.keys():
                tf_writer.add_scalar('eval_' + loss_type, eval_loss[loss_type], iteration)

            _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            print_info = '\n'.join([key + ":" + str(eval_score[key]) for key in eval_score.keys()])
            logger.info('\nValidation results of iter {}:\n'.format(iteration) + print_info)
            logger.info('\noverall score of iter {}: {}\n'.format(iteration, current_score))
            val_result_history[epoch] = {'eval_score': eval_score}
            logger.info('Save model at iter {} to {}.'.format(iteration, checkpoint_path))

            logger.info('Current best model details:')
            print_info = 'DVC:\n' + '\n'.join([key + ":" + str(best_dvc_details[key]) for key in best_dvc_details.keys()])
            logger.info(print_info)


            # save the model parameter and  of best epoch
            if len(best_suffix_list):
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': vars(opt),
                                      'iter': iteration,
                                      'epoch': best_epoch,
                                      'best_val_score': best_val_score,
                                      'best_dvc_score':best_dvc_score,
                                      'best_localization_score':best_localization_score,
                                      'best_pc_score':best_pc_score,
                                      'result_json_path': result_json_path,
                                      'avg_proposal_num': eval_score['avg_proposal_number'],
                                      'Precision': eval_score['Precision'],
                                      'Recall': eval_score['Recall']
                                      }

                # suffix = "RL" if sc_flag else "CE"
                torch.save(saved_pth, os.path.join(save_folder, 'model-best.pth'))
                for best_suffix in best_suffix_list:
                    torch.save(saved_pth, os.path.join(save_folder, 'model-best-{}.pth'.format(best_suffix)))
                logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))

            saved_info['last'] = {'opt': vars(opt),
                                  'iter': iteration,
                                  'epoch': epoch,
                                  'best_val_score': best_val_score,
                                  'best_dvc_score':best_dvc_score,
                                  'best_localization_score':best_localization_score,
                                  'best_pc_score':best_pc_score
                                  }
            saved_info['history'] = {'val_result_history': val_result_history,
                                     'loss_history': loss_history,
                                     'lr_history': lr_history,
                                     # 'query_matched_fre_hist': query_matched_fre_hist,
                                     }
            with open(os.path.join(save_folder, 'info.json'), 'w') as f:
                json.dump(saved_info, f)
            logger.info('Save info to info.json')

            model.train()

        epoch += 1
        lr_scheduler.step()
        if need_update_text_encoder and opt.text_encoder_learning_strategy == 'multi_step':
            text_encoder_scheduler.step()
        torch.cuda.empty_cache()
        # Stop criterion
        if epoch >= opt.epoch:
            tf_writer.close()
            break

    return saved_info


if __name__ == '__main__':
    opt = opts.parse_opts()
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # to avoid OMP problem on macos
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    train(opt)
