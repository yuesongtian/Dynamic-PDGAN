export PATH=/usr/local/cuda/bin/:$PATH
#python3.6 /apdcephfs/share_1290939/yuesongtian/stylegan2/run_training.py --num-gpus=4 --data-dir=/apdcephfs/share_1290939/yuesongtian/datasets/ --config=config-f --dataset=church --mirror-augment=true --total-kimg=15000 --result-dir /apdcephfs/share_1290939/yuesongtian/stylegan2_results

#python3.6 run_distillation.py --num-gpus=8 \
#    --config=config-f \
#    --dataset=ffhq --data-dir=/apdcephfs/share_1367250/yuesongtian/ \
#    --mirror-augment=true --total-kimg=60000 \
#    --result-dir /apdcephfs/private_yuesongtian/stylegan2_results/ \
#    --run_name ffhq_1-1_G_AlignInternalOutput_coeff1
#--data-dir=/apdcephfs/share_1367250/0_public_datasets/LSUN/


#for network in /apdcephfs/share_1367250/yuesongtian/stylegan2_results/1-1ch_ffhq_dgl/network-snapshot-047511.pkl 
#do
#  python3.6 run_metrics.py --data-dir=/apdcephfs/share_1367250/yuesongtian --network=$network \
#    --metrics=fid50k,ppl_wend,ppl2_wend --dataset=ffhq --mirror-augment=true
#done

for network in /apdcephfs/share_1367250/yuesongtian/stylegan2_results/ffhq_1-1_G_AlignInternalOutput/network_29366_2.5155.pkl /apdcephfs/share_1367250/yuesongtian/pretrained_models/stylegan2-ffhq-config-f.pkl
do 
  python3.6 run_generator.py generate-images --network=$network \
    --seeds=5000-5100 --truncation-psi=0.5 --result-dir=/apdcephfs_cq2/share_1367250/yuesongtian/stylegan2_results
done
