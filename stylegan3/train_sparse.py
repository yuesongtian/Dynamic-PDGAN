# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop, training_sparse_loop
from training import hessian_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    #training_loop.training_loop(rank=rank, **c)
    training_sparse_loop.training_loop(rank=rank, **c)
    #hessian_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run, experiment_name):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    # And scan over the output directory, find the latest checkpoint.
    c.run_dir = os.path.join(outdir, experiment_name)
    if os.path.exists(c.run_dir):
        cur_kimg = 0
        cur_pkl = ''
        for item in os.listdir(c.run_dir):
            if 'network-snapshot' in item and 'pkl' in item:
                kimg = int(item.split('-')[-1].rstrip('.pkl'))
                if kimg > cur_kimg:
                    cur_kimg = kimg
                    cur_pkl = item
        if cur_kimg > 0:
            c.resume_kimg = cur_kimg
            c.resume_pkl = os.path.join(c.run_dir, cur_pkl)
        

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Experiment name:     {experiment_name}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    if not os.path.exists(c.run_dir):
        os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--experiment_name',help='The name of the directory of log file and checkpoints', metavar='STR', required=True)
@click.option('--dbar_path',help='The name of the directory of the pretrained model', metavar='STR', required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--data_full',    help='Training data', metavar='[ZIP|DIR]',                      type=str, required=False)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)
@click.option('--dbar_lambda',  help='V(G,Dbar) weight', metavar='FLOAT',                       type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Sparse hyperparameters.
@click.option('--sema',         help='Whether to use sparse ema', metavar='BOOL',               type=bool, default=False, show_default=True)
@click.option('--dy_mode',      help='mode of exploration', metavar='STR',                      type=str, default='GD', show_default=True)
@click.option('--sparse_init',  help='sparse initialization', metavar='STR',                    type=str, default='ERK', show_default=True)
@click.option('--G_growth',     help='Growth mode for G. Choose from: momentum, random, random_unfired, and gradient.', metavar='STR', type=str, default='gradient', show_default=True)
@click.option('--D_growth',     help='Growth mode for D. Choose from: momentum, random, random_unfired, and gradient.', metavar='STR', type=str, default='random', show_default=True)
@click.option('--death',        help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.', metavar='STR', type=str, default='magnitude', show_default=True)
@click.option('--redistribution', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.', metavar='STR', type=str, default='none', show_default=True)
@click.option('--death_rate',   help='The pruning rate / death rate.', metavar='FLOAT', type=float, default=0.50, show_default=True)
@click.option('--density',      help='The density of the overall sparse network.', metavar='FLOAT', type=float, default=0.3, show_default=True)
@click.option('--update_frequency', help='how many iterations to train between parameter exploration', metavar='INT', type=int, default=2000, show_default=True)
@click.option('--decay_schedule', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.', metavar='STR', type=str, default='cosine', show_default=True)
@click.option('--densityG',     help='The density ratio of G.', metavar='FLOAT', type=float, default=0.05, show_default=True)
@click.option('--densityD',     help='The density ratio of D.', metavar='FLOAT', type=float, default=0.05, show_default=True)
@click.option('--imbalanced',   help='Enable imbalanced training mode. Default: False.', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--pruning_mode', help='pruning mode: uniform_G, uniform_GD, global_G and global_GD', metavar='STR', type=str, default='', show_default=True)
@click.option('--pruning_rate', help='Pruning rate', metavar='FLOAT', type=float, default=None, show_default=True)
@click.option('--multiplier',   help='multiplier for extended training', metavar='INT', type=int, default=1, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    #c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=64, w_dim=128, mapping_kwargs=dnnlib.EasyDict())
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    #c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2_search.SuperDiscriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    #c.D_kwargs = dnnlib.EasyDict(class_name='training.ViT.ViT_scale3_local_new_rp.Discriminator', args=dnnlib.EasyDict())
    #c.D_kwargs.args.d_window_size=4
    #c.D_kwargs.args.df_dim=16
    #c.D_kwargs.args.d_depth=3
    #c.D_kwargs.args.patch_size=16
    #c.D_kwargs.args.d_norm='ln'
    #c.D_kwargs.args.d_act='gelu'
    #c.D_kwargs.args.img_size=1024
    #c.D_kwargs.args.diff_aug='translation,erase_ratio,color' 
    c.Dbar_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.Dbar_paths =  None
                    #[opts.dbar_path]
                    #'/apdcephfs/share_1367250/yuesongtian/pretrained_models/stylegan2-afhqv2-512x512.pkl']
                    #'/apdcephfs/share_1367250/yuesongtian/pretrained_models/stylegan2-ffhq-1024x1024.pkl']
                    #'/apdcephfs_cq2/share_1367250/yuesongtian/stylegan2_results/stylegan2_pretrain_var/network_best_54600_2.4261.pkl']
                    #'/apdcephfs/share_1367250/yuesongtian/pretrained_models/stylegan3-t-afhqv2-512x512.pkl']
                    #'/apdcephfs/share_1367250/yuesongtian/pretrained_models/cifar10.pkl']
                    #'/apdcephfs/share_1367250/yuesongtian/stylegan2_results/stylegan2_c10_ada/network-final.pkl'
    c.channel_fractions = [1]
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    #c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss_ens', Dbar_lambda=0.1)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    # Masking kwargs
    print(f'opts is {opts}')
    c.mask_kwargs = dnnlib.EasyDict(class_name='sparselearning.core.Masking', G_growth=opts.g_growth, D_growth=opts.d_growth,
                                    dy_mode=opts.dy_mode,
                                    death_rate=opts.death_rate,
                                    multiplier=opts.multiplier,
                                    resume=opts.resume,
                                    sparse_init=opts.sparse_init,
                                    update_frequency=opts.update_frequency,
                                    density=opts.density,
                                    densityG=opts.densityg, densityD=opts.densityd,
                                    imbalanced=opts.imbalanced)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    # For few-shot synthesis, you need compute FID w.r.t. the complete dataset.
    c.full_training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data_full)
    c.full_training_set_kwargs.use_labels = opts.cond
    c.full_training_set_kwargs.xflip = opts.mirror


    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = opts.cbase
    c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = opts.cmax
    c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    #c.D_kwargs.block_kwargs.kernels = [1,3,5,3,5,7]
    #c.D_kwargs.block_kwargs.operation_type = ['common', 'common', 'common', 'sep', 'sep', 'sep']
    #c.D_kwargs.block_kwargs.sample_type = 'random_continuous'
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    #c.loss_kwargs.Dbar_lambda = opts.dbar_lambda
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.experiment_name = opts.experiment_name

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:  # opts.mbstd_group:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        #c.G_kwargs.class_name = 'pg_modules.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = 0
        c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = None
        c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run, experiment_name=opts.experiment_name)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------