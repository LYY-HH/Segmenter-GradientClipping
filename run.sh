##############train#################
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=12131 DATASET=/data/yingyueli/data WORK=./ python -m segm.train \
--log-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_weaktr_pseudo_mask_ms_crf \
--dataset pascal_context --backbone deit_small_patch16_224 --decoder mask_transformer \
--batch-size 4 --epochs 100 -lr 1e-4 \
--num-workers 2 --eval-freq 1 \
--ann-dir /data/yingyueli/WeakTr_official/WeakTr_results/weaktr/pseudo-mask-ms-crf \
--pus-type local_adaptive --pus-beta 1.2 --pus-kernel 120


##############val##################
#####multi-scale#####
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12313 \
segm/eval/miou.py --window-batch-size 1 --backbone multi --multiscale --weight 0.8 \
pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms_multi \
pascal_context
#######add crf#######
python -m segm.eval.make_crf \
--list data/voc12/ImageSets/Segmentation/val.txt \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms_multi \
--predict-png-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_pred_ms_multi \
--img-path data/voc12/JPEGImages \
--gt-folder data/voc12/SegmentationClassAug \

##############test##################
#####multi-scale#####
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12313 \
segm/eval/miou.py --window-batch-size 1 --backbone multi --multiscale --weight 0.8 \
--eval-split ImageSets/Segmentation/test.txt \
pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms_test_multi \
pascal_context
#######add crf#######
python -m segm.eval.make_crf \
--list data/voc12/ImageSets/Segmentation/test.txt \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms_test_multi \
--predict-png-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_pred_ms_test_multi \
--img-path data/voc12/JPEGImages \


#############generate mask###############
#####multi-scale#####
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --backbone multi --weight 0.8 \
--eval-split ImageSets/Segmentation/train.txt \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms_train_multi \
pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
pascal_context
#######add crf#######
python -m segm.eval.make_crf \
--list data/voc12/ImageSets/Segmentation/train.txt \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms_train_multi \
--predict-png-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms_train_multi \
--img-path data/voc12/JPEGImages \
--gt-folder data/voc12/SegmentationClassAug \
