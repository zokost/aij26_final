"""Stage-one ConvNeXt-L DINOv3 / UPerNet: three scene-grouped folds."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import Images, Loss, frames, seed, validation_batch

THRESHOLDS = (.05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55,
              .60, .65, .70, .75, .80, .85, .90, .95, .97, .99)
MIN_AREAS = (0., .0005, .001, .002, .005, .01)
ENCODER = "tu-convnext_large.dinov3_lvd1689m"


def teacher(pretrained=False):
    return smp.UPerNet(encoder_name=ENCODER, encoder_weights="imagenet" if pretrained else None,
                       in_channels=3, classes=1)


def split(root, fold):
    # The shared outer validation is never included in any teacher's training.
    train, val = frames(root, val_fraction=.12)
    original, mask = train.orgl_img_path.astype(str), train.gt_path.astype(str)
    groups = np.where(~original.isin(["", "nan"]), original,
                      np.where(~mask.isin(["", "nan"]), mask, "solo_" + train.index.astype(str)))
    folds = np.array([int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16) % 3 for key in groups])
    held, train = train[folds == fold].copy(), train[folds != fold].copy()
    val = val.sample(n=min(8000, len(val)), random_state=42)
    return train, held, val


def score(intersections, areas, targets, pixels):
    """Original threshold/area sweep, including the original >1% FPR rule."""
    intersections, areas = np.stack(intersections), np.stack(areas)
    targets, pixels = np.asarray(targets), np.asarray(pixels)
    positive, rows = targets > 0, []
    for fraction in MIN_AREAS:
        keep = areas >= (fraction * pixels)[:, None] if fraction else np.ones_like(areas, dtype=bool)
        inter, area = np.where(keep, intersections, 0.), np.where(keep, areas, 0.)
        dice = ((2 * inter + 1e-7) / (area + targets[:, None] + 1e-7))[positive].mean(0)
        fpr = (area[~positive] > (.01 * pixels[~positive])[:, None]).mean(0)
        tnr = 1 - fpr
        aic = np.where(dice + tnr > 0, 2 * dice * tnr / (dice + tnr + 1e-7), 0.)
        for index, threshold in enumerate(np.asarray(THRESHOLDS, np.float32)):
            rows.append(dict(thr=float(threshold), min_area_frac=fraction, dice=float(dice[index]),
                             fpr=float(fpr[index]), aic=float(aic[index])))
    best = max(rows, key=lambda row: row["aic"])
    return dict(best, n_pos=int(positive.sum()), n_neg=int((~positive).sum()))


@torch.inference_mode()
def evaluate(net, loader):
    net.eval()
    thresholds = torch.tensor(THRESHOLDS, dtype=torch.float32, device="cuda")
    intersections, areas, targets, pixels = [], [], [], []
    for images, masks in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(images.cuda(non_blocking=True))
        for index, mask in enumerate(masks):
            target = mask.cuda(non_blocking=True).bool()
            probability = F.interpolate(logits[index:index + 1].float(), target.shape,
                                        mode="bilinear", align_corners=False).sigmoid()[0, 0]
            prediction = probability[None] >= thresholds[:, None, None]
            intersections.append((prediction & target).sum((1, 2)).float().cpu().numpy())
            areas.append(prediction.sum((1, 2)).float().cpu().numpy())
            targets.append(float(target.sum()))
            pixels.append(target.numel())
    return score(intersections, areas, targets, pixels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(3), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/stage1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-epoch", type=int, default=40, help="Keeps the original 40-epoch LR schedule")
    args = parser.parse_args()
    if not 1 <= args.stop_epoch <= 40:
        parser.error("stop-epoch must be between 1 and 40")
    seed()
    torch.set_num_threads(4)
    train, held, val = split(args.data, args.fold)
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
        held.to_csv(args.out / "heldout.csv", index=False)
    print(json.dumps(dict(fold=args.fold, train=len(train), heldout=len(held), validation=len(val))), flush=True)
    loader = DataLoader(Images(train, args.data, True, size=832), batch_size=6, shuffle=True,
                        drop_last=True, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    validation = DataLoader(Images(val, args.data, False, size=832), batch_size=6,
                            num_workers=args.workers, pin_memory=True, collate_fn=validation_batch)
    net = teacher(pretrained=not args.resume).cuda()
    encoder_ids = {id(param) for param in net.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) in encoder_ids], "lr": 3e-5},
        {"params": [p for p in net.parameters() if id(p) not in encoder_ids], "lr": 3e-4}], weight_decay=1e-4)
    total, warmup = 40 * len(loader), max(1, int(40 * len(loader) * .05))
    def factor(step):
        return step / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    criterion = Loss(tversky_beta=None, positive_balance=False)
    history, start, best_aic, best_epoch = [], 1, -1., 0
    if args.resume:
        saved = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        if saved["fold"] != args.fold:
            raise ValueError("Resume fold differs")
        net.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, start = saved["history"], saved["epoch"] + 1
        best_aic, best_epoch = saved["best_aic"], saved["best_epoch"]
        del saved
    cfg = dict(arch="upernet", encoder=ENCODER, encoder_weights="imagenet", img_size=832,
               decoder_channels=None, squish=True, seed=42, epochs=40, batch_size=6,
               lr=3e-4, encoder_lr_mult=.1, weight_decay=1e-4, warmup_frac=.05)
    for epoch in range(start, args.stop_epoch + 1):
        net.train()
        running, count = 0., 0
        optimizer.zero_grad(set_to_none=True)
        for images, targets in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(net(images.cuda(non_blocking=True)), targets.cuda(non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.detach().item() * len(images)
            count += len(images)
        metric = evaluate(net, validation)
        history.append(dict(epoch=epoch, loss=running / count, **metric))
        saved = dict(model=net.state_dict(), cfg=cfg, fold=args.fold, epoch=epoch, best=metric)
        if metric["aic"] > best_aic:
            best_aic, best_epoch = metric["aic"], epoch
            torch.save(saved, args.out / "best.pt")
        saved.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history,
                     best_aic=best_aic, best_epoch=best_epoch)
        temporary = args.out / "last.tmp"
        torch.save(saved, temporary)
        temporary.replace(args.out / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(history[-1]), flush=True)
        if epoch - best_epoch >= 12:
            break


if __name__ == "__main__":
    main()
