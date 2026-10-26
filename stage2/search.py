"""Sequential NB-only architecture/hyperparameter screening with a fixed control split."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from .recipe import resolve
from .runtime import COLUMNS, read_rows
from .validate import DEFAULT_CSV, check_disjoint, validation_rows


def candidates(plan):
    """Validate a named comparison plan and resolve every variant explicitly."""
    if (not isinstance(plan, dict) or set(plan) != {"defaults", "variants"}
            or not isinstance(plan["defaults"], dict)
            or not isinstance(plan["variants"], list) or not plan["variants"]):
        raise ValueError("Plan needs defaults and a nonempty variants list")
    seen, result = set(), []
    for item in plan["variants"]:
        if not isinstance(item, dict) or set(item) != {"name", "overrides"} or not isinstance(item["overrides"], dict):
            raise ValueError("Each variant needs name and overrides")
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("Run names must be unique safe path components")
        seen.add(name)
        result.append((name, resolve({**plan["defaults"], **item["overrides"]})))
    return result


def best_epoch(history):
    # Rank by positive/negative tradeoff, then fewer false alarms, then earlier epoch.
    return min(history, key=lambda r: (-r["nb"]["aic"], r["nb"]["fpr_negative"], r["epoch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).parent / "assets/search_space.json")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=Path("weights/FULL2_late_avg5.pt"))
    parser.add_argument("--nb-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Audit inputs and write commands; no GPU training")
    args = parser.parse_args()
    variants = candidates(json.loads(args.plan.read_text()))
    if any(c["initialization"] == "checkpoint" for _, c in variants) and not args.initial.is_file():
        raise FileNotFoundError(args.initial)
    if len({(c["batch"], c["schedule_epochs"], c["final_epoch"], c["seed"], c["img_size"]) for _, c in variants}) != 1:
        raise ValueError("Within a screening plan fix batch, epochs, seed and resolution")
    full = validation_rows(args.nb_csv, args.root)
    # Audit the entire control panel against every manifest used by this plan.
    epochs = max(c["schedule_epochs"] for _, c in variants)
    files = [args.manifests / "stage2_train.csv", *[args.manifests / f"epoch{e:02d}.csv" for e in range(1, epochs + 1)]]
    training = pd.concat([read_rows(p, args.root)[COLUMNS] for p in files], ignore_index=True)
    check_disjoint(full, training)
    raw = pd.read_csv(args.nb_csv, keep_default_na=False)
    # Preserve edit-type proportions in both panels instead of letting a small
    # control split accidentally lose a rare manipulation category.
    strata = raw.edit_type if "edit_type" in raw else None
    tune, control = train_test_split(raw, test_size=.25, random_state=42, stratify=strata)
    # Source identities must stay together even if a custom CSV repeats a scene.
    check_disjoint(read_rows_frame(tune, args.root), read_rows_frame(control, args.root))
    signature = dict(plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                     nb_sha256=hashlib.sha256(args.nb_csv.read_bytes()).hexdigest(),
                     manifests={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                     root=str(args.root.resolve()), initial=str(args.initial.resolve()))
    state = args.out / "search.json"
    if args.out.exists():
        if not args.resume or not state.exists() or json.loads(state.read_text()) != signature:
            raise ValueError("Existing output needs --resume and identical plan/data inputs")
    else:
        args.out.mkdir(parents=True)
        state.write_text(json.dumps(signature, indent=2) + "\n")
    tune.to_csv(args.out / "tune.csv", index=False)
    control.to_csv(args.out / "control.csv", index=False)
    summaries, commands = [], []
    for name, cfg in variants:
        config_path, run = args.out / f"{name}.json", args.out / name
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        command = [sys.executable, "-m", "stage2.train", "student", "--root", str(args.root),
                   "--manifests", str(args.manifests), "--recipe", str(config_path),
                   "--nb-val-csv", str(args.out / "tune.csv"), "--out", str(run)]
        if cfg["initialization"] == "checkpoint":
            command += ["--initial", str(args.initial)]
        commands.append(dict(name=name, command=command, recipe=cfg))
        (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if args.dry_run:
            continue
        history_path = run / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        if not history or history[-1]["epoch"] < cfg["final_epoch"]:
            if run.exists():
                command += ["--resume"]
            with (args.out / f"{name}.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            history = json.loads(history_path.read_text())
        best = best_epoch(history)
        summaries.append(dict(name=name, epoch=best["epoch"], **best["nb"],
                              checkpoint=str(run / f"epoch{best['epoch']}.pt"), **cfg))
        pd.DataFrame(summaries).sort_values(["aic", "fpr_negative", "epoch"], ascending=[False, True, True]).to_csv(
            args.out / "summary.csv", index=False)
    (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    if args.dry_run:
        print(f"Audited {len(variants)} variants; tune={len(tune)}, control={len(control)}; no training started")
        return
    # The control split is evaluated only after fixing the winning configuration/epoch.
    winner = min(summaries, key=lambda r: (-r["aic"], r["fpr_negative"], r["epoch"], r["name"]))
    selected_path = args.out / "selected.json"
    if selected_path.exists() and json.loads(selected_path.read_text()) != winner:
        raise ValueError("Frozen winner changed: create a new study instead of reusing the control split")
    selected_path.write_text(json.dumps(winner, indent=2) + "\n")
    control_path = args.out / "control_metrics.json"
    if not control_path.exists():
        subprocess.run([sys.executable, "-m", "stage2.validate", "--checkpoint", winner["checkpoint"],
                        "--root", str(args.root), "--csv", str(args.out / "control.csv"),
                        "--out", str(control_path)], check=True)
    print(json.dumps(dict(selected=winner, control=json.loads(control_path.read_text())), indent=2))


def read_rows_frame(frame, root):
    result = frame.copy()
    for column in COLUMNS:
        result[column] = result[column].map(lambda p: str(root / p) if p else "")
    return result


if __name__ == "__main__":
    main()
