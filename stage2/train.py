"""Train a cross-fit teacher or the fixed KDALL student recipe."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import Loss, seed
from .runtime import COLUMNS, Images, companions, model, read_rows, worker_seed
from .validate import DEFAULT_CSV, check_disjoint, evaluate_nb, validation_rows


def epoch_frame(base, manifests, root, epoch, teacher, fold):
    path = manifests / (f"teacher_extra{fold}.csv" if teacher else f"epoch{epoch:02d}.csv")
    extra = read_rows(path, root)
    clean = extra[extra.no_src != 1] if "no_src" in extra else extra
    frame = pd.concat([base, extra[COLUMNS], companions(clean)], ignore_index=True)
    expected = 52598 if teacher else 23835
    if len(frame) != expected:
        raise ValueError(f"Unexpected row count: {len(frame)} instead of {expected}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["teacher", "student"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(3), default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-epoch", type=int)
    parser.add_argument("--nb-val-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    seed()
    torch.set_num_threads(4)
    teacher = args.phase == "teacher"
    size, batch, workers, schedule, final = (832, 6, 10, 12, 12) if teacher else (640, 18, 8, 40, 32)
    stop = final if args.stop_epoch is None else args.stop_epoch
    if not 1 <= stop <= schedule:
        parser.error("stop-epoch is outside the fixed LR schedule")
    net = model(teacher)
    saved = torch.load(args.initial, map_location="cpu", weights_only=False, mmap=True)
    if teacher:
        net.load_state_dict(saved["model"], strict=True)
    else:
        torch.nn.Module.load_state_dict(net.encoder, {k[len("encoder."):]: v for k, v in saved["model"].items()
                                                      if k.startswith("encoder.")}, strict=True)
    del saved
    seed()
    base = read_rows(args.manifests / "stage2_train.csv", args.root)[COLUMNS]
    validation = validation_rows(args.nb_val_csv, args.root)
    if len(base) != 19837:
        raise ValueError("Expected the fixed 19837-row stage2 training split")
    dataset = Images(epoch_frame(base, args.manifests, args.root, 1, teacher, args.fold), size, soft=not teacher)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        pin_memory=True, worker_init_fn=worker_seed, generator=generator,
                        persistent_workers=False)
    if not teacher and len(loader) != 1324:
        raise ValueError("Expected 1324 student optimizer steps per epoch")
    net.cuda()
    encoder_ids = {id(p) for p in net.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) in encoder_ids], "lr": 1e-5},
        {"params": [p for p in net.parameters() if id(p) not in encoder_ids], "lr": 1e-4}], weight_decay=1e-4)
    total, warmup = schedule * len(loader), max(1, int(schedule * len(loader) * .05))
    def factor(step):
        return step / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    criterion = Loss()
    history, start = [], 1
    if args.resume:
        saved = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        if saved["phase"] != args.phase or saved["fold"] != args.fold:
            raise ValueError("Resume recipe differs")
        net.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["generator"])
        history, start = saved["history"], saved["epoch"] + 1
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_cuda"])
        del saved
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    cfg = dict(arch="upernet" if teacher else "segformer",
               encoder="tu-convnext_large.dinov3_lvd1689m" if teacher else "mit_b2",
               decoder_channels=None if teacher else 320, img_size=size, squish=True)
    for epoch in range(start, stop + 1):
        dataset.frame = epoch_frame(base, args.manifests, args.root, epoch, teacher, args.fold).reset_index(drop=True)
        check_disjoint(validation, dataset.frame)
        net.train()
        total_loss, count = 0., 0
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
            total_loss += float(loss.detach()) * len(images)
            count += len(images)
        # Validation is deterministic, uses FP32 and does not choose the fixed final epoch.
        nb = evaluate_nb(net, validation, size)
        history.append(dict(epoch=epoch, loss=total_loss / count, nb=nb))
        compact = dict(model=net.state_dict(), cfg=cfg, epoch=epoch, phase=args.phase, fold=args.fold)
        torch.save(compact, args.out / f"epoch{epoch}.pt")
        compact.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history,
                       generator=generator.get_state(), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state())
        temporary = args.out / "last.tmp"
        torch.save(compact, temporary)
        temporary.replace(args.out / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
