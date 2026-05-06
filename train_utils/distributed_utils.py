import torch
import torch.distributed as dist
import sys
from tqdm import tqdm


def reduce_value(value, average=True):
    """多 GPU 训练时同步 Loss 等数值"""
    world_size = get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function):
    """训练一个 Epoch 的标准流程"""
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    # 进度条展示
    pbar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(pbar):
        images, masks = data
        outputs = model(images.to(device))["out"]

        loss = loss_function(outputs, masks.to(device))
        loss.backward()

        accu_loss += loss.detach()
        pbar.desc = f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}"

        if not torch.isfinite(loss):
            print(f"WARNING: non-finite loss, ending training {loss}")
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    """验证集评估流程"""
    model.eval()
    metric_logger = {"Dice": 0, "IoU": 0}

    pbar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(pbar):
        images, masks = data
        outputs = model(images.to(device))["out"]

        # 这里调用我们刚才写的 metrics.py 计算指标
        from .metrics import SegmentationMetric
        metric = SegmentationMetric().calculate_all(outputs, masks.to(device))

        metric_logger["Dice"] += metric["Dice"]
        metric_logger["IoU"] += metric["IoU"]

        pbar.desc = f"[val epoch {epoch}] Dice: {metric_logger['Dice'] / (step + 1):.3f}"

    return metric_logger["Dice"] / len(data_loader)