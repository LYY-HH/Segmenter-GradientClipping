##############train#################
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=12131 DATASET=/data/yingyueli/data WORK=./ \
python -m segm.train --log-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf \
--dataset coco --backbone deit_small_patch16_224 --decoder mask_transformer \
--batch-size 4 --epochs 100 -lr 1e-4 \
--num-workers 2 --eval-freq 1 \
--ann-dir /data/yingyueli/WeakTr_official/WeakTr_results_coco/weaktr_coco/pseudo-mask-ms-crf \
--pus-type local_adaptive --pus-beta 1.2 --pus-kernel 120

##############val_mini##################
#####multi-scale#####
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --eval-split voc_format/val_mini.txt \
pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/checkpoint.pth \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_prob_ms \
coco

#######add crf#######
python -m segm.eval.make_crf \
--list data/coco/voc_format/val_mini.txt \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_prob_ms \
--img-path data/coco/images \
--gt-folder data/coco/voc_format/class_labels \
--num-cls 91 --dataset coco


##############val##################
#####multi-scale#####

#COCO在eval时中间可能会被kill掉，通过val_tmp筛选未生成的图片重新生成
python val_tmp.py --predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_prob_ms
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --eval-split voc_format/val_tmp.txt \
pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/checkpoint.pth \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_prob_ms \
coco

#######add crf#######
python -m segm.eval.make_crf \
--list data/coco/voc_format/val.txt \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_prob_ms \
--predict-png-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_pred_ms \
--img-path data/coco/images \
--gt-folder data/coco/voc_format/class_labels \
--num-cls 91 --dataset coco

python evaluation.py --list data/coco/voc_format/val.txt \
--gt-dir data/coco/voc_format/class_labels \
--predict-dir /data/yingyueli/GradientClipping/pus1.2_adaptive_seg_deit_small_patch16_224_mask_lr3e-4_WeakTrCOCOPseudoMask/seg_pred_ms \
--num-classes 91 \

##############generate mask##################
#####multi-scale#####

#COCO在eval时中间可能会被kill掉，通过val_tmp筛选未生成的图片重新生成
python val_tmp.py --predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_prob_ms \
--list data/coco/voc_format/train.txt
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --eval-split voc_format/val_tmp.txt \
pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/checkpoint.pth \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_prob_ms_train \
coco

#######add crf#######
python -m segm.eval.make_crf \
--list data/coco/voc_format/train.txt \
--predict-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_prob_ms_train \
--predict-png-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_COCO_weaktr_coco_pseudo_mask_ms_crf/seg_pred_ms_train \
--img-path data/coco/images \
--gt-folder data/coco/voc_format/class_labels \
--num-cls 91 --dataset coco

python evaluation.py --list data/coco/voc_format/train.txt \
--gt-dir data/coco/voc_format/class_labels \
--predict-dir /data/yingyueli/GradientClipping/pus1.2_adaptive_seg_deit_small_patch16_224_mask_lr3e-4_WeakTrCOCOPseudoMask/seg_pred_train \
--num-classes 91 \
