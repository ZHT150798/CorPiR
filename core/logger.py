import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def _resolve_scale(opt):
    """Resolve the up-sampling factor and every field that follows from it.

    Values written explicitly in the config always win; this only fills in the
    ones that were left out, so that a single config can serve x2 / x4 / x8.
    """
    scale = opt.get('scale')
    if scale is None:
        raise ValueError("The config must define 'scale' (the up-sampling factor).")
    scale = int(scale)
    opt['scale'] = scale

    # substitute {scale} in the experiment name, the paths and the data roots
    if isinstance(opt.get('name'), str):
        opt['name'] = opt['name'].replace('{scale}', str(scale))
    for key, value in opt['path'].items():
        if isinstance(value, str):
            opt['path'][key] = value.replace('{scale}', str(scale))
    for dataset_opt in opt['datasets'].values():
        if isinstance(dataset_opt.get('dataroot'), str):
            dataset_opt['dataroot'] = dataset_opt['dataroot'].replace('{scale}', str(scale))

        r_res = dataset_opt.get('r_resolution')
        if r_res is None:
            raise ValueError("Each dataset section needs 'r_resolution'.")
        if r_res % scale != 0:
            raise ValueError(
                'r_resolution ({}) must be divisible by scale ({}).'.format(r_res, scale))
        if dataset_opt.get('l_resolution') is None:
            dataset_opt['l_resolution'] = r_res // scale

    unet_opt = opt['model']['unet']
    diffusion_opt = opt['model']['diffusion']
    val_opt = opt['datasets']['val']

    if diffusion_opt.get('image_size') is None:
        diffusion_opt['image_size'] = val_opt['r_resolution']
    if unet_opt.get('attn_res') is None:
        unet_opt['attn_res'] = [val_opt['l_resolution']]

    # the UNet halves the resolution once per entry in channel_multiplier
    image_size = diffusion_opt['image_size']
    mults = unet_opt['channel_multiplier']
    levels = [image_size // (2 ** i) for i in range(len(mults))]
    attn_res = unet_opt['attn_res'][0]
    if attn_res not in levels:
        raise ValueError(
            'attn_res={} is not a resolution level of the UNet. With image_size={} and '
            'channel_multiplier={} the available levels are {}.'.format(
                attn_res, image_size, mults, levels))

    # UNet.__init__ sizes the CConstrainer as `512 // attn_res[0] * 16`, which only
    # coincides with the true UNet width at that level for the released configuration.
    expected = unet_opt['inner_channel'] * mults[levels.index(attn_res)]
    if expected != 512 // attn_res * 16:
        raise ValueError(
            'Unsupported UNet configuration: the CConstrainer width is hard-coded as '
            '512 // attn_res * 16 = {}, but this UNet is {} channels wide at resolution {}. '
            'Only inner_channel=64 with channel_multiplier=[1,2,4,8,8] and image_size=128 '
            'is currently supported.'.format(512 // attn_res * 16, expected, attn_res))

    return opt


def parse(args):
    phase = args.phase
    opt_path = args.config
    gpu_ids = args.gpu_ids
    enable_wandb = args.enable_wandb
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # resolve the up-sampling factor and everything derived from it
    opt = _resolve_scale(opt)

    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    experiments_root = os.path.join(
        'experiments', '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    # change dataset length limit
    opt['phase'] = phase

    # export CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    # debug
    if 'debug' in opt['name']:
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3
        opt['datasets']['train']['batch_size'] = 2
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['model']['beta_schedule']['val']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 6
        opt['datasets']['val']['data_len'] = 3

    # validation in train phase
    if phase == 'train':
        opt['datasets']['val']['data_len'] = 3

    # W&B Logging
    try:
        log_wandb_ckpt = args.log_wandb_ckpt
        opt['log_wandb_ckpt'] = log_wandb_ckpt
    except:
        pass
    try:
        log_eval = args.log_eval
        opt['log_eval'] = log_eval
    except:
        pass
    try:
        log_infer = args.log_infer
        opt['log_infer'] = log_infer
    except:
        pass
    opt['enable_wandb'] = enable_wandb
    
    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
