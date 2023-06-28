export TORCH_HOME=/apdcephfs/share_1367250/yuesongtian/pretrained_models/CA-GAN
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

#python3.6 calc_metrics.py \
#          --network=/apdcephfs/private_yuesongtian/stylegan2_results/stylegan2_ffhq_G-0.3_D-0.5_exploreGD_noAug/network_60000_2.6358.pkl \
#          --metrics=fid50k_full

#python3.6 gen_images.py \
#          --network="/apdcephfs/private_yuesongtian/stylegan2_results/stylegan2_ffhq_G-0.5_D-0.5_exploreGD_noAug/network_58200_3.0973.pkl" \
#          --seeds=6000-6002 \
#          --outdir=/apdcephfs_cq2/share_1367250/yuesongtian/stylegan2_results/stylegan2_church_samples/

python3.6 train.py --outdir=/apdcephfs/private_yuesongtian/stylegan2_results/ \
    --experiment_name stylegan3_church_DyPDGAN_ada \
    --dbar_path /apdcephfs/private_yuesongtian/ft_local/church_stylegan3_ada_2.8291.pkl \
    --cfg=stylegan2 --data=/apdcephfs/private_yuesongtian/datasets/church-256x256.zip \
    --data_full=/apdcephfs/private_yuesongtian/datasets/church-256x256.zip \
    --dbar_lambda=0.1 \
    --kimg=60000 \
    --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=ada
#--sema=True --dy_mode=GD --densityG=0.1 --densityD=0.5 --G_growth=gradient --D_growth=random --update_frequency=2000 --imbalanced=True \
#/apdcephfs/share_1367250/0_public_datasets/ffhq/ffhq-1024x1024.zip \
#LSUN/church/church-256x256.zip
#AFHQ/afhqv2-512x512.zip

#python3.6 train_sparse.py --outdir=/apdcephfs_cq2/share_1367250/yuesongtian/stylegan2_results/ \
#    --experiment_name stylegan2_c10_G-0.9_D-0.1_exploreGD_updateOnce_MaskResumeSteps_noaug \
#    --dbar_path giaogiao_ \
#    --sema=True --dy_mode=GD --densityG=0.1 --densityD=0.9 --G_growth=gradient --D_growth=random --update_frequency=2000 --imbalanced=True \
#    --cfg=stylegan2 --data=/apdcephfs/share_1367250/yuesongtian/ft_local/cifar10.zip \
#    --data_full=/apdcephfs/share_1367250/yuesongtian/ft_local/cifar10.zip \
#    --dbar_lambda=0.1 \
#    --kimg=100000 \
#    --gpus=1 --batch=64 --gamma=0.01 --mirror=1 --aug=noaug

#python3.6 train.py --outdir=/apdcephfs_cq2/share_1367250/yuesongtian/stylegan2_results/ \
#    --experiment_name stylegan2_c10_base_noaug \
#    --glr 0.0025 --dlr 0.0025 \
#    --map-depth 2 \
#    --dbar_path giaogiao_ \
#    --kimg 100000 \
#    --cfg=stylegan2_c10 --data=/apdcephfs/share_1367250/yuesongtian/ft_local/cifar10.zip \
#    --data_full=/apdcephfs/share_1367250/yuesongtian/ft_local/cifar10.zip \
#    --dbar_lambda=0.6 \
#    --gpus=1 --batch=64 --gamma=0.01 --mirror=1 --aug=noaug
