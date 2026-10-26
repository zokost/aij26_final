"""Train a cross-fit teacher or the fixed NC17 student recipe."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import Loss, seed
from .runtime import COLUMNS, Images, companions, model, read_rows, worker_seed
from .validate import DEFAULT_CSV, check_disjoint, evaluate_nb, validation_rows
from .recipe import DEFAULT, load as load_recipe
from .distillation import DistillationImages, DistillationLoss, target_index


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
    parser.add_argument("phase", choices=["teacher", "stage2-teacher", "student"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(3), default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-epoch", type=int)
    parser.add_argument("--nb-val-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--recipe", type=Path, help="Student research overrides; omitted = fixed NC17")
    parser.add_argument('--stage2-targets',type=Path,help='Source-held-out stage2 logit index; default manifests/stage2_oof_index.csv')
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    seed(recipe["seed"])
    torch.set_num_threads(4)
    teacher = args.phase != "student"
    source_teacher = args.phase == 'stage2-teacher'
    if teacher and args.recipe:
        parser.error("Research overrides apply to students only")
    pretrained = not teacher and recipe["initialization"] == "imagenet"
    if not pretrained and args.initial is None:
        parser.error("Checkpoint initialization requires --initial")
    if pretrained and args.initial is not None:
        parser.error("ImageNet initialization must not also supply --initial")
    size, batch, workers, schedule, final = (832, 6, 10, 12, 12) if teacher else (
        recipe["img_size"], recipe["batch"], 8, recipe["schedule_epochs"], recipe["final_epoch"])
    if source_teacher:workers=8
    stop = final if args.stop_epoch is None else args.stop_epoch
    if not 1 <= stop <= schedule:
        parser.error("stop-epoch is outside the fixed LR schedule")
    net = model(teacher, recipe["encoder"], recipe["decoder_channels"], pretrained)
    if pretrained:
        # B1/B2 have the same feature-channel interface. Copy a common, independently
        # seeded head so encoder size cannot change decoder initialization via RNG use.
        seed(recipe["seed"])
        reference = model(decoder_channels=recipe["decoder_channels"])
        net.decoder.load_state_dict(reference.decoder.state_dict(), strict=True)
        net.segmentation_head.load_state_dict(reference.segmentation_head.state_dict(), strict=True)
        del reference
    else:
        saved = torch.load(args.initial, map_location="cpu", weights_only=False, mmap=True)
        if teacher:
            net.load_state_dict(saved["model"], strict=True)
        else:
            # A new decoder is intentional: transfer only the stage-one representation.
            torch.nn.Module.load_state_dict(net.encoder, {k[len("encoder."):]: v for k, v in saved["model"].items()
                                                          if k.startswith("encoder.")}, strict=True)
        del saved
    seed(recipe["seed"])
    base_path=args.manifests/(f'teacher_stage2_fold{args.fold}.csv' if source_teacher else 'stage2_train.csv')
    base = read_rows(base_path, args.root)[COLUMNS]
    validation = validation_rows(args.nb_val_csv, args.root)
    if source_teacher:
        held=read_rows(args.manifests/f'stage2_heldout_fold{args.fold}.csv',args.root)
        check_disjoint(held,base)
    elif len(base) != 19837:
        raise ValueError("Expected the fixed 19837-row stage2 training split")
    def rows(epoch):
        return base if source_teacher else epoch_frame(base,args.manifests,args.root,epoch,teacher,args.fold)
    distilling=not teacher and recipe['stage2_kd_weight']>0
    target_path=args.stage2_targets or args.manifests/'stage2_oof_index.csv'
    dataset_kwargs=dict(soft=not teacher,augmentation=recipe['augmentation'],seed_value=recipe['seed'])
    if distilling:dataset_kwargs['teacher_index']=target_index(target_path,args.root,base)
    dataset=(DistillationImages if distilling else Images)(rows(1),size,**dataset_kwargs)
    generator = torch.Generator().manual_seed(recipe["seed"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        pin_memory=True, worker_init_fn=worker_seed, generator=generator,
                        persistent_workers=False)
    if not teacher and recipe["batch"] == 18 and len(loader) != 1324:
        raise ValueError("Expected 1324 student optimizer steps per epoch")
    net.cuda()
    encoder_ids = {id(p) for p in net.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) in encoder_ids], "lr": recipe["lr"] * recipe["encoder_lr_mult"]},
        {"params": [p for p in net.parameters() if id(p) not in encoder_ids], "lr": recipe["lr"]}], weight_decay=1e-4)
    total, warmup = schedule * len(loader), max(1, int(schedule * len(loader) * .05))
    def factor(step):
        return step / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    loss_kwargs=dict(tversky_beta=recipe['tversky_beta'],positive_balance=recipe['positive_balance'])
    criterion=DistillationLoss(weight=recipe['stage2_kd_weight'],**loss_kwargs) if distilling else Loss(**loss_kwargs)
    history, start = [], 1
    if args.resume:
        saved = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        if saved["phase"] != args.phase or saved["fold"] != args.fold:
            raise ValueError("Resume recipe differs")
        if saved.get("recipe", None if teacher else DEFAULT) != (None if teacher else recipe):
            raise ValueError("Cannot resume with different research hyperparameters")
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
    # File hashes and versions make a run auditable without embedding data or weights.
    manifest_paths = [base_path, args.nb_val_csv]
    manifest_paths += ([args.manifests/f'stage2_heldout_fold{args.fold}.csv'] if source_teacher else
                      [args.manifests / f"teacher_extra{args.fold}.csv"] if teacher else
                       [args.manifests / f"epoch{e:02d}.csv" for e in range(1, schedule + 1)])
    if distilling:manifest_paths.append(target_path)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_paths}
    metadata = dict(recipe=None if teacher else recipe, manifests=hashes,
                    python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    steps_per_epoch=len(loader), validation_threshold=.75,
                    initialization=str(args.initial) if args.initial else "ImageNet via SMP")
    metadata_path = args.out / "run.json"
    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["manifests"] != hashes:
            raise ValueError("Cannot resume with modified data/validation manifests")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    cfg = dict(arch="upernet" if teacher else "segformer",
               encoder="tu-convnext_large.dinov3_lvd1689m" if teacher else recipe["encoder"],
               decoder_channels=None if teacher else recipe["decoder_channels"], img_size=size, squish=True)
    for epoch in range(start, stop + 1):
        dataset.frame = rows(epoch).reset_index(drop=True)
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
        history.append(dict(epoch=epoch, loss=total_loss / count, nb=nb,
                            learning_rates=scheduler.get_last_lr()))
        compact = dict(model=net.state_dict(), cfg=cfg, epoch=epoch, phase=args.phase, fold=args.fold,
                       recipe=None if teacher else recipe)
        if distilling:compact['distillation']=dict(weight=recipe['stage2_kd_weight'],index_sha256=hashes[target_path.name])
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
