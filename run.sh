CUDA_VISIBLE_DEVICES=0 MASTER_PORT=12131 DATASET=/data/yingyueli/data WORK=./ python -m segm.train --log-dir ep50_seg_deit_small_distilled_patch16_224_mask_lr1e-4_WeakTrCOCOPseudoMask \
--dataset coco --backbone deit_small_distilled_patch16_224 --decoder mask_transformer \
--batch-size 4 --epochs 50 -lr 1e-4 \
--num-workers 2 --eval-freq 1 \
--ann-dir /data/yingyueli/WeakTrCOCOPseudoMask \
--resume --run-id 4f6983ace34a43ac8248391929227918  \

CUDA_VISIBLE_DEVICES=0 MASTER_PORT=12131 DATASET=/data/yingyueli/data WORK=./ python -m segm.train --log-dir layerdecay0.8_ep50_seg_deit_small_distilled_patch16_224_mask_lr1e-4_WeakTrCOCOPseudoMask \
--dataset coco --backbone deit_small_distilled_patch16_224 --decoder mask_transformer \
--batch-size 4 --epochs 50 -lr 1e-4 \
--num-workers 2 --eval-freq 1 --layer-decay 0.8 \
--ann-dir /data/yingyueli/WeakTrCOCOPseudoMask \
--resume --run-id 35ee192839724077b5fc258270fa88e1  \

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12121 \
segm/eval/miou.py --window-batch-size 1 \
pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
pascal_context

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12313 \
segm/eval/miou.py --window-batch-size 1 --multiscale \
pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
pascal_context

# 生成multiscale npy文件
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --eval-split voc_format/val_mini.txt \
--predict-dir lr1e-4_seg_vit_small_patch16_384_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms \
lr1e-4_seg_vit_small_patch16_384_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/checkpoint.pth \
coco

python val_tmp.py
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --eval-split voc_format/val_tmp.txt \
--predict-dir pus1.2_adaptive_lr1e-4_seg_vit_small_patch16_384_COCO_MCTformerV2SE@coco@\
train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms \
pus1.2_adaptive_lr1e-4_seg_vit_small_patch16_384_COCO_MCTformerV2SE@coco@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
coco

# 生成multiscale npy文件
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --eval-split voc_format/val_tmp.txt\
--predict-dir pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_\
MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms \
pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
pascal_context

# 生成multiscale npy文件
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --eval-split voc_format/val_mini.txt \
seg_deit_small_patch16_224_mask_lr1e-4_WeakTrCOCOPseudoMask/checkpoint.pth \
--predict-dir seg_deit_small_patch16_224_mask_lr1e-4_WeakTrCOCOPseudoMask/seg_prob_ms \
coco

# 做crf并测试
python -m segm.eval.make_crf \
--list data/coco/voc_format/val.txt \
--predict-dir seg_deit_small_patch16_224_mask_lr1e-4_WeakTrCOCOPseudoMask/seg_prob_ms \
--img-path data/coco/images \
--gt-folder data/coco/voc_format/class_labels \
--num-cls 91 --dataset coco --type png


CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12919 \
segm/eval/miou.py --crf --predict-dir pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_\
MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms -frac-dataset 0.01 \
--img-path /data/yingyueli/data/pcontext/VOCdevkit/VOC2012/JPEGImages \
pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
pascal_context

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12919 \
segm/eval/miou.py --window-batch-size 1 --multiscale \
--predict-dir pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_\
MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms \
--img-path /data/yingyueli/data/pcontext/VOCdevkit/VOC2012/JPEGImages \
pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
pascal_context

CUDA_VISIBLE_DEVICES=2 MASTER_PORT=12131 DATASET=/data/yingyueli/data WORK=./ python -m segm.train \
--log-dir pus1.2_local_adaptive_kernel120_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42 \
--dataset pascal_context --backbone deit_small_patch16_224 --decoder mask_transformer \
--batch-size 4 --epochs 100 -lr 1e-4 \
--num-workers 2 --eval-freq 1 \
--ann-dir /data/yingyueli/MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42 \
--pus-type local_adaptive --pus-beta 1.2 --pus-kernel 120


CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. MASTER_PORT=12901 DATASET=/data/zhulianghui/data WORK=./ \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12919 \
segm/train.py --log-dir pus1.2_local_adaptive_kernel120_ep50_seg_vit_small_patch16_384_mask_lr1e-4_WeakTrCOCOPseudoMask \
--dataset coco --backbone vit_small_patch16_384 --decoder mask_transformer \
--batch-size 4 --epochs 50 -lr 1e-4 \
--num-workers 2 --eval-freq 1 \
--ann-dir /data/yingyueli/WeakTrCOCOPseudoMask --pus-type local_adaptive \
--pus-beta 1.2 --pus-kernel 120 --resume --run-id 735531c179ea4830b0e909604a3b55fa


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. DATASET=/data/yingyueli/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12313 \
segm/eval/miou.py --window-batch-size 1 --multiscale \
--predict-dir pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE\
@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms \
pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42\
/checkpoint.pth \
pascal_context

python val_tmp.py --predict-dir pus1.2_adaptive_lr1e-4_seg_deit_small_patch16_224_mask_VOC2012_MCTformerV2SE\
@train@scale=1.0,0.8,1.2@aff_fg=0.41_bg=0.42/seg_prob_ms
