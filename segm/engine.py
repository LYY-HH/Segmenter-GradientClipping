import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL, weight_reduce_loss
import segm.utils.torch as ptu

import torch.nn.functional as F


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    pus_type=None,
    pus_beta=None,
    pus_power=None,
    pus_kernel=None,
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="none")
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        with amp_autocast():
            seg_pred = model.forward(im)
            loss = criterion(seg_pred, seg_gt)

            logger.update(
                mean_loss=loss.mean().item(),
            )
            if pus_type == "adaptive" and loss.mean() < pus_beta:
                mean_loss = loss.detach().clone().mean()
                loss = torch.clamp(loss, 0, mean_loss)
            elif pus_type == "local_adaptive" and loss.mean() < pus_beta:
                detach_loss = loss.detach().clone()
                mean_loss = detach_loss.mean()
                b, h, w = detach_loss.shape
                detach_loss = detach_loss.mean(dim=0).unsqueeze(0)
                local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=pus_kernel,
                                          stride=pus_kernel, padding=h % pus_kernel,
                                          count_include_pad=False).squeeze(1)
                local_mean = torch.maximum(local_mean, mean_loss)
                local_mean = torch.repeat_interleave(local_mean, b, dim=0)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=1)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=2)

                clamp_loss = loss - local_mean
                clamp_loss = torch.clamp(clamp_loss, None, 0)
                loss = clamp_loss + local_mean
            elif pus_type == "min_local_adaptive" and loss.mean() < pus_beta:
                detach_loss = loss.detach().clone()
                mean_loss = detach_loss.mean()
                b, h, w = detach_loss.shape
                detach_loss = detach_loss.mean(dim=0).unsqueeze(0)
                local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=pus_kernel,
                                          stride=pus_kernel, padding=h % pus_kernel,
                                          count_include_pad=False).squeeze(1)
                local_mean = torch.minimum(mean_loss, local_mean)
                local_mean = torch.repeat_interleave(local_mean, b, dim=0)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=1)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=2)

                clamp_loss = loss - local_mean
                clamp_loss = torch.clamp(clamp_loss, None, 0)
                loss = clamp_loss + local_mean
            elif pus_type == "mean_local_adaptive" and loss.mean() < pus_beta:
                detach_loss = loss.detach().clone()
                mean_loss = detach_loss.mean()
                b, h, w = detach_loss.shape
                detach_loss = detach_loss.mean(dim=0).unsqueeze(0)
                local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=pus_kernel,
                                          stride=pus_kernel, padding=h % pus_kernel,
                                          count_include_pad=False).squeeze(1)
                local_mean = (local_mean + mean_loss) / 2
                local_mean = torch.repeat_interleave(local_mean, b, dim=0)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=1)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=2)

                clamp_loss = loss - local_mean
                clamp_loss = torch.clamp(clamp_loss, None, 0)
                loss = clamp_loss + local_mean
            elif pus_type == "adaptive_power" and loss.mean() < pus_beta:
                mean_loss = loss.detach().clone().mean()
                loss = torch.clamp(loss, 0, mean_loss ** pus_power)
            elif pus_type == "adaptive_slide":
                detach_loss = loss.detach().clone()
                b, h, w = detach_loss.shape
                local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=pus_kernel,
                                          stride=pus_kernel, padding=h % pus_kernel,
                                          count_include_pad=False).squeeze(1)

                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=1)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=2)
                clamp_loss = (loss - local_mean)
                clamp_loss = torch.clamp(clamp_loss, None, 0)
                loss = clamp_loss + local_mean
            elif pus_type == "global_adaptive_slide" and loss.mean() < pus_beta:
                detach_loss = loss.detach().clone()
                b, h, w = detach_loss.shape
                local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=pus_kernel,
                                          stride=pus_kernel, padding=h % pus_kernel,
                                          count_include_pad=False).squeeze(1)

                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=1)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=2)
                clamp_loss = (loss - local_mean)
                clamp_loss = torch.clamp(clamp_loss, None, 0)
                loss = clamp_loss + local_mean
            elif pus_type == "local_adaptive_slide":
                detach_loss = loss.detach().clone()
                b, h, w = detach_loss.shape
                local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=pus_kernel,
                                          stride=pus_kernel, padding=h % pus_kernel,
                                          count_include_pad=False).squeeze(1)

                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=1)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=2)
                for_clamp_loss = (loss - local_mean)
                clamp_loss = for_clamp_loss.clone()
                clamp_loss[for_clamp_loss > 0] *= (pus_beta - local_mean[for_clamp_loss > 0])
                clamp_loss = torch.clamp(clamp_loss, None, 0)
                clamp_loss[for_clamp_loss > 0] /= (pus_beta - local_mean[for_clamp_loss > 0])
                loss = clamp_loss + local_mean
            elif pus_type == "batch_adaptive_slide":
                detach_loss = loss.detach().clone()
                b, h, w = detach_loss.shape
                detach_loss = detach_loss.mean(dim=0).unsqueeze(0)
                local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=pus_kernel,
                                          stride=pus_kernel, padding=h % pus_kernel,
                                          count_include_pad=False).squeeze(1)

                local_mean = torch.repeat_interleave(local_mean, b, dim=0)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=1)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=2)
                clamp_loss = loss - local_mean
                clamp_loss = torch.clamp(clamp_loss, None, 0)
                loss = clamp_loss + local_mean
            elif pus_type == "batch_local_adaptive_slide":
                detach_loss = loss.detach().clone()
                b, h, w = detach_loss.shape
                detach_loss = detach_loss.mean(dim=0).unsqueeze(0)
                local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=pus_kernel,
                                          stride=pus_kernel, padding=h % pus_kernel,
                                          count_include_pad=False).squeeze(1)

                local_mean = torch.repeat_interleave(local_mean, b, dim=0)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=1)
                local_mean = torch.repeat_interleave(local_mean, pus_kernel, dim=2)
                for_clamp_loss = (loss - local_mean)
                clamp_loss = for_clamp_loss.clone()
                clamp_loss[for_clamp_loss > 0] *= (pus_beta - local_mean[for_clamp_loss > 0])
                clamp_loss = torch.clamp(clamp_loss, None, 0)
                clamp_loss[for_clamp_loss > 0] /= (pus_beta - local_mean[for_clamp_loss > 0])
                loss = clamp_loss + local_mean
            loss = weight_reduce_loss(
                loss, weight=None, reduction="mean", avg_factor=None)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            enc_learning_rate=optimizer.param_groups[0]["lr"],
            dec_learning_rate=optimizer.param_groups[1]["lr"],
        )

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger, scores['mean_iou']
