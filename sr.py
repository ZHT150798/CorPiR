import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
from torchvision.transforms import functional as trans_fn
import os
import numpy as np
import lpips
from PIL import Image
import os
import cv2
import core.imgproc as imgproc
from core.image_quality_assessment import PSNR, SSIM, set_seeds




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train_CorPiR.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='1')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-use_ddim', default=False)
    #parser.add_argument('-upscale', type=int, default=4)

    set_seeds(seed=3407)
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Initialize the sharpness evaluation function
    psnr_model = PSNR(0, False)
    ssim_model = SSIM(0, False)
    loss_fn = lpips.LPIPS(net='vgg', spatial=True)
    mse_model = torch.nn.L1Loss(reduction='sum')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        max_psnr = -1e5
        min_lpips = 1e5
        while current_step < n_iter:
            current_epoch += 1
            scaler = torch.cuda.amp.GradScaler()

            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data, "train")
                diffusion.optimize_parameters(scaler)
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    diffusion.print_lr()
                    avg_psnr, avg_psnr_liif = 0.0, 0.0
                    avg_ssim, avg_ssim_liif = 0.0, 0.0
                    avglpips, avglpips_liif = 0.0, 0.0
                    consistency = 0.0

                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False, use_ddim=False)
                        visuals = diffusion.get_current_visuals()
                        xcon_img = visuals['CON']
                        lr_img = visuals['LR']  # uint8
                        batch_size, channels, lr_image_height, lr_image_width = lr_img.shape
                        shape = [batch_size,
                                 round(xcon_img.shape[2]),
                                 round(xcon_img.shape[3]),
                                 channels]
                        hr_img = visuals['HR'].view(*shape).permute(0, 3, 1, 2).contiguous()  # uint8
                        fake_img = visuals['INF']  # uint8
                        xcon_img = imgproc.tensor_to_image(xcon_img, False, False)
                        xcon_img = cv2.cvtColor(xcon_img, cv2.COLOR_RGB2BGR)
                        lr_img = imgproc.tensor_to_image(lr_img, False, False)
                        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
                        hr_img = imgproc.tensor_to_image(hr_img, False, False)
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
                        fake_img = imgproc.tensor_to_image(fake_img, False, False)
                        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
                        sr_img = imgproc.tensor_to_image(visuals['SR'], False, False)
                        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

                        # generation
                        cv2.imwrite('{}/{}_{}_hr.png'.format(result_path, current_step, idx), hr_img)
                        cv2.imwrite('{}/{}_{}_lr.png'.format(result_path, current_step, idx), lr_img)
                        cv2.imwrite('{}/{}_{}_inf.png'.format(result_path, current_step, idx), fake_img)
                        cv2.imwrite('{}/{}_{}_con.png'.format(result_path, current_step, idx), xcon_img)
                        cv2.imwrite('{}/{}_{}_sr.png'.format(result_path, current_step, idx), sr_img)
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        gt_tensor = visuals['HR'].view(*shape).permute(0, 3, 1, 2).contiguous()
                        sr_dtensor = trans_fn.resize(visuals['SR'].unsqueeze(0), (lr_image_width, lr_image_width), Image.BICUBIC)
                        avg_psnr += psnr_model(visuals['SR'].unsqueeze(0).clamp(0, 1), gt_tensor).item()
                        avg_ssim += ssim_model(visuals['SR'].unsqueeze(0).clamp(0, 1), gt_tensor).item()
                        avglpips += loss_fn.forward(visuals['SR'].unsqueeze(0).clamp(0, 1), gt_tensor).mean().item()
                        scale = 128//lr_image_height
                        consistency += (mse_model(sr_dtensor.clamp(0, 1) * 255.0, visuals['LR'].clamp(0, 1) * 255.0) / 1e5 * scale**2)
                        avg_psnr_liif += psnr_model(visuals['CON'], gt_tensor).item()
                        avg_ssim_liif += ssim_model(visuals['CON'], gt_tensor).item()
                        avglpips_liif += loss_fn.forward(visuals['CON'], gt_tensor).mean().item()

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    consistency = consistency / idx
                    avg_psnr_liif = avg_psnr_liif / idx
                    avg_ssim_liif = avg_ssim_liif / idx
                    avglpips = avglpips /idx
                    avglpips_liif = avglpips_liif / idx
                    if avglpips <= min_lpips:  # and dist.get_rank() == 0:
                        min_lpips = avglpips
                        diffusion.save_network(current_epoch, current_step, best='lpips_{}'.format(round(min_lpips, 4)))
                        if avg_psnr >= max_psnr :
                            max_psnr = avg_psnr
                            diffusion.save_network(current_epoch, current_step, best='best_lpips_{}_psnr_{}'.format(round(min_lpips, 4), round(max_psnr,3)))
                    elif avg_psnr >= max_psnr :  # and dist.get_rank() == 0:
                        max_psnr = avg_psnr
                        diffusion.save_network(current_epoch, current_step, best='psnr_{}'.format(round(max_psnr,3)))

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # LPIPS: {:.4e}'.format(avglpips))
                    logger.info('# Validation # Consistency: {:.4e}'.format(consistency))

                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info(
                        '<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, lpips: {:.4e}, consistency: {:.4e}, ssim: {:.4e},'.format(
                            current_epoch, current_step, avg_psnr, avglpips, consistency, avg_ssim))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr, avg_psnr_liif = 0.0, 0.0
        avg_ssim, avg_ssim_liif = 0.0, 0.0
        avg_lpips, avg_lpips_liif = 0.0, 0.0
        avg_consistency, avg_consistency_liif = 0.0, 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True, use_ddim=args.use_ddim)
            visuals = diffusion.get_current_visuals()

            xcon_img = visuals['CON']
            lr_img = visuals['LR']  # uint8
            batch_size, channels, lr_image_height, lr_image_width = lr_img.shape
            shape = [batch_size,
                     round(xcon_img.shape[2]),
                     round(xcon_img.shape[3]),
                     channels]
            hr_img = visuals['HR'].view(*shape).permute(0, 3, 1, 2).contiguous()  # uint8
            fake_img = visuals['INF']  # uint8
            xcon_img = imgproc.tensor_to_image(xcon_img, False, False)
            xcon_img = cv2.cvtColor(xcon_img, cv2.COLOR_RGB2BGR)
            lr_img = imgproc.tensor_to_image(lr_img, False, False)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
            hr_img = imgproc.tensor_to_image(hr_img, False, False)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
            fake_img = imgproc.tensor_to_image(fake_img, False, False)
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]),
                        '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_process, sr_img = Metrics.tensor2img(visuals['SR']), imgproc.tensor_to_image(visuals['SR'][-1],
                                                                                                False, False)
                sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                sr_process = cv2.cvtColor(sr_process, cv2.COLOR_RGB2BGR)
                cv2.imwrite('{}/{}_{}_sr_process.png'.format(result_path, current_step, idx), sr_process)
                cv2.imwrite('{}/{}_{}_sr.png'.format(result_path, current_step, idx), sr_img)  # uint8

            cv2.imwrite('{}/{}_{}_hr.png'.format(result_path, current_step, idx), hr_img)
            cv2.imwrite('{}/{}_{}_lr.png'.format(result_path, current_step, idx), lr_img)
            cv2.imwrite('{}/{}_{}_inf.png'.format(result_path, current_step, idx), fake_img)
            cv2.imwrite('{}/{}_{}_con.png'.format(result_path, current_step, idx), xcon_img)

            # generation
            lrh = opt["datasets"]["val"]["l_resolution"]
            sr_tensor, gt_tensor, lr_tensor = visuals['SR'][-1].unsqueeze(0), visuals['HR'].view(*shape).permute(0, 3,
                                                                                                                 1,
                                                                                                                 2).contiguous(), \
                                              visuals['LR']
            sr_dtensor = trans_fn.resize(sr_tensor, (lrh, lrh), Image.BICUBIC)

            lpips_metrics = loss_fn.forward(sr_tensor.clamp(0, 1), gt_tensor).mean().item()
            scale = 128 // lr_image_height
            consistency = (mse_model(sr_dtensor.clamp(0, 1) * 255.0, lr_tensor * 255.0) / 1e5 * scale**2)
            eval_psnr = psnr_model(sr_tensor.clamp(0, 1), gt_tensor).item()
            eval_ssim = ssim_model(sr_tensor.clamp(0, 1), gt_tensor).item()


            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            avg_lpips += lpips_metrics
            avg_consistency += consistency


            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr,
                                           eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_psnr_liif = avg_psnr_liif / idx
        avg_ssim_liif = avg_ssim_liif / idx
        avg_lpips_liif = 0 if avg_lpips_liif / idx < 0 else avg_lpips_liif / idx
        avg_lpips = 0 if avg_lpips / idx < 0 else avg_lpips / idx
        avg_consistency = avg_consistency / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # LPIPS: {:.4e}'.format(avg_lpips))
        logger.info('# Validation # Consistency: {:.4e}'.format(avg_consistency))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info(
            '<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}, lpips: {:.4e}, consistency：{:.4e}'.format(
                current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips, avg_consistency))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
