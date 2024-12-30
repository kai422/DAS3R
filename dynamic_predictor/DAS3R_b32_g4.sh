# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3"  

# Run the Python script using torchrun (adjust if using distributed training)
torchrun --nproc_per_node=4 --master_port=27777 launch.py  --mode=train \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', \
                        img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder_and_3d_predictor')" \
    --train_dataset="10_000 @ PointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='../data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)" \
    --test_dataset="1 * PointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='../data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777) + 1 * SintelDUSt3R(dset='final', z_far=80, S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 224)], seed=777)" \
    --train_criterion="ConfLoss(Regr3D_MMask(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv_MMask(L21, gt_scale=True)" \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr=0.00005 --min_lr=1e-06 --warmup_epochs=3 --epochs=50 --batch_size=8 --accum_iter=1 \
    --test_batch_size=8 \
    --save_freq=3 --keep_freq=5 --eval_freq=50  \
    --output_dir=results/MSeg_from_monst3r_b32_g4 \
    --num_workers=16 --wandb \

