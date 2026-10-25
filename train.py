"""Original two-GPU stage-one training and full-labelled finetuning."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from common import Images, Loss, evaluate, frames, model, seed, validation_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["base", "finetune"])
    parser.add_argument("--data", type=Path, default=Path("data/stage1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="epoch161.pt for finetuning")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stop-epoch", type=int, help="Stopping does not shorten the LR schedule")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("Use python -m torch.distributed.run --nproc_per_node=2 train.py ...")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, device = dist.get_rank(), torch.device("cuda", local)
    torch.set_num_threads(4)
    seed()
    base = args.phase == "base"
    schedule_epochs, final_epoch, batch, lr = (200, 161, 6, 3e-4) if base else (10, 5, 40, 3e-5)
    stop = args.stop_epoch or final_epoch
    if not 1 <= stop <= schedule_epochs:
        parser.error("stop-epoch must be inside the fixed LR schedule")
    train, val = frames(args.data)
    if (len(train), len(val)) != (134055, 4210):
        raise ValueError("Unexpected official stage-one data or split")
    digest = hashlib.sha256("\n".join(sorted(val.chng_img_path.astype(str))).encode()).hexdigest()
    # Phase two must preserve the original train-then-validation row ordering.
    frame = train if base else pd.concat([train, val], ignore_index=True)
    dataset = Images(frame, args.data, training=True)
    sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=0 if base else 42,
                                 shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    validation = DataLoader(Images(val, args.data, training=False), batch_size=batch,
                            num_workers=args.workers, pin_memory=True, collate_fn=validation_batch) if base else None
    if not base and len(loader) != 1728:
        raise ValueError("Expected 1728 optimizer steps per full-data epoch")
    net = model(pretrained=base).to(device)
    if not base:
        if args.checkpoint is None:
            parser.error("finetune requires --checkpoint")
        initial = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(initial["model"], strict=True)
        del initial
    encoder_ids = {id(p) for p in net.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) in encoder_ids], "lr": lr * .1},
        {"params": [p for p in net.parameters() if id(p) not in encoder_ids], "lr": lr}], weight_decay=1e-4)
    total, warmup = len(loader) * schedule_epochs, max(1, int(len(loader) * schedule_epochs * .05))
    def factor(step):
        return step / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    start, history = 1, []
    if args.resume:
        saved = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        if saved["phase"] != args.phase:
            raise ValueError("Resume phase differs")
        net.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, history = saved["epoch"] + 1, saved["history"]
        del saved
    elif rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    if rank == 0:
        print(json.dumps({"phase": args.phase, "train": len(frame), "validation": len(val) if base else 0,
                          "validation_paths_sha256": digest, "schedule_epochs": schedule_epochs,
                          "steps_per_epoch": len(loader), "effective_batch": batch * 2}), flush=True)
    wrapped = DistributedDataParallel(net, device_ids=[local])
    criterion = Loss()
    for epoch in range(start, stop + 1):
        sampler.set_epoch(epoch)
        wrapped.train()
        running = torch.zeros(2, device=device)
        optimizer.zero_grad(set_to_none=True)
        for images, target in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(wrapped(images.to(device, non_blocking=True)), target.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += torch.stack([loss.detach() * len(images), torch.tensor(len(images), device=device)])
        dist.all_reduce(running)
        if rank == 0:
            row = {"epoch": epoch, "loss": (running[0] / running[1]).item()}
            if base:
                row.update(evaluate(net, validation, device))
            history.append(row)
            checkpoint = {"model": net.state_dict(), "epoch": epoch, "phase": args.phase,
                          "cfg": {"arch": "segformer", "encoder": "mit_b2", "img_size": 704,
                                  "decoder_channels": 128, "squish": True}}
            torch.save(checkpoint, args.out / f"epoch{epoch}.pt")
            checkpoint.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history)
            temporary = args.out / "last.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.out / "last.pt")
            (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
