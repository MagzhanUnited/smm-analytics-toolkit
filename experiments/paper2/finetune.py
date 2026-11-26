"""Fine-tune a multilingual transformer on KazSAnDRA and run the same evaluation battery as the baselines.

    python -I finetune.py --task p3 --name xlmr_p3_full_s1 --seed 1 --epochs 3 [--train-size 4000]

Standard recipe without a hyperparameter search: AdamW, learning rate 2e-5, linear warm-up over 6 % of
the steps and linear decay, weight decay 0.01, gradient clipping at 1.0, plain cross-entropy. The epoch
is chosen by validation macro-F1. Outputs go to ``<runs-dir>/<name>/``; a finished run is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are read from a local directory
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()


class Classifier:
    """A tokenizer and a sequence-classification model with length-bucketed batching."""

    def __init__(self, model_dir: Path, k: int, max_len: int, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=k)
        self.max_len = max_len
        self.device = torch.device(device)
        self.model.to(self.device)

    def encode(self, texts: list) -> list:
        return self.tokenizer(list(texts), truncation=True, max_length=self.max_len)["input_ids"]

    def collate(self, ids: list) -> dict:
        # widths are rounded up to a multiple of 8: few distinct tensor shapes keep the MPS memory pool small
        width = min(self.max_len, -(-max(len(row) for row in ids) // 8) * 8)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, row in enumerate(ids):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1
        return {"input_ids": input_ids.to(self.device), "attention_mask": mask.to(self.device)}

    @torch.inference_mode()
    def logits(self, texts: list, batch_size: int = 256) -> np.ndarray:
        self.model.eval()
        ids = self.encode(texts)
        order = np.argsort([len(row) for row in ids], kind="stable")
        out = np.empty((len(ids), self.model.config.num_labels), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            out[chunk] = self.model(**self.collate([ids[i] for i in chunk])).logits.float().cpu().numpy()
        self.release()
        return out

    def release(self) -> None:
        """Return cached GPU buffers to the system (the MPS allocator keeps them otherwise)."""
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def predict(self, texts: list) -> np.ndarray:
        return self.logits(texts).argmax(1)

    def to(self, device: str) -> None:
        self.device = torch.device(device)
        self.model.to(self.device)


def training_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list:
    """Shuffled batches of similar length: sort inside chunks of 50 batches, then shuffle the batches."""
    index = rng.permutation(len(lengths))
    chunk = batch_size * 50
    index = np.concatenate(
        [
            part[np.argsort(lengths[part], kind="stable")]
            for part in (index[i : i + chunk] for i in range(0, len(index), chunk))
        ]
    )
    batches = [index[i : i + batch_size] for i in range(0, len(index), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=("pc", "p3"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--text-col", choices=("text", "text_cleaned"), default="text")
    parser.add_argument("--model-dir", type=Path, default=C.ML_ROOT / "models" / "xlm-roberta-base")
    parser.add_argument("--data-dir", type=Path, default=C.ML_ROOT / "data" / "kazsandra")
    parser.add_argument("--runs-dir", type=Path, default=C.ML_ROOT / "runs")
    parser.add_argument("--train-size", type=int, default=0, help="0 uses the whole training set")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--probe", type=int, default=1000)
    parser.add_argument("--no-battery", action="store_true", help="test metrics only (learning-curve points)")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="balanced: n / (k * class count) in the loss",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="stop early; for smoke tests only")
    args = parser.parse_args(argv)

    run_dir = args.runs_dir / args.name
    if (run_dir / "metrics.json").exists():
        print("already done:", args.name)
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frames = C.load_task(args.data_dir, args.task, args.text_col)
    labels = C.LABELS[args.task]
    k = len(labels)
    train = (
        frames["train"]
        if args.train_size == 0
        else C.stratified_sample(frames["train"], args.train_size, args.seed)
    )
    valid = C.valid_for_size(frames["valid"], len(train))
    test = frames["test"]
    y_valid, y_test = valid["y"].to_numpy(), test["y"].to_numpy()
    valid_texts, test_texts = valid["x"].tolist(), test["x"].tolist()

    clf = Classifier(args.model_dir, k, args.max_len, args.device)
    ids = clf.encode(train["x"].tolist())
    lengths = np.array([len(row) for row in ids])
    y_train = torch.tensor(train["y"].to_numpy(), dtype=torch.long)
    steps_per_epoch = -(-len(ids) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.06 * total_steps))
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {
            "params": [p for n, p in clf.model.named_parameters() if not any(s in n for s in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in clf.model.named_parameters() if any(s in n for s in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimiser = torch.optim.AdamW(groups, lr=args.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (
            (step + 1) / warmup
            if step < warmup
            else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        ),
    )
    print(f"{args.name}: train={len(ids)} valid={len(valid)} steps/epoch={steps_per_epoch} device={args.device} "
          f"tokens mean={lengths.mean():.1f} p99={np.percentile(lengths, 99):.0f} truncated={(lengths >= args.max_len).mean():.4f}", flush=True)  # fmt: skip

    counts = np.bincount(train["y"], minlength=k)
    weight = (
        None
        if args.class_weight == "none"
        else torch.tensor(len(train) / (k * counts), dtype=torch.float32, device=clf.device)
    )
    history, best, stale, step, train_seconds, eval_seconds = [], None, 0, 0, 0.0, 0.0
    stop = False
    for epoch in range(1, args.epochs + 1):
        clf.model.train()
        start = time.perf_counter()
        running, seen = torch.zeros((), device=clf.device), 0
        for batch_index in training_batches(lengths, args.batch, rng):
            batch = clf.collate([ids[i] for i in batch_index])
            loss = torch.nn.functional.cross_entropy(
                clf.model(**batch).logits.float(), y_train[batch_index].to(clf.device), weight=weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad(set_to_none=True)
            step += 1
            running += loss.detach() * len(batch_index)
            seen += len(batch_index)
            if step % 250 == 0:
                clf.release()
            if step % 500 == 0:
                print(
                    f"  step {step}/{total_steps} loss {running.item() / seen:.4f} {time.perf_counter() - start:.0f}s in epoch",
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        epoch_seconds = time.perf_counter() - start
        train_seconds += epoch_seconds
        start = time.perf_counter()
        valid_pred = clf.predict(valid_texts)
        eval_seconds += time.perf_counter() - start
        s = C.scores(C.confusion(y_valid, valid_pred, k))
        row = {"epoch": epoch, "train_loss": round(running.item() / max(1, seen), 4), "valid_macro_f1": round(s["macro_f1"], 4),
               "valid_accuracy": round(s["accuracy"], 4), "train_seconds": round(epoch_seconds, 1)}  # fmt: skip
        history.append(row)
        print("  ", row, flush=True)
        if not np.isfinite(row["train_loss"]):
            raise RuntimeError("training diverged")
        if best is None or s["macro_f1"] > best[0]:
            best = (
                s["macro_f1"],
                epoch,
                {name: value.detach().to("cpu", copy=True) for name, value in clf.model.state_dict().items()},
            )
            stale = 0
        else:
            stale += 1
        clf.release()
        if stop or stale >= args.patience:
            break
    clf.model.load_state_dict(best[2])
    clf.to(args.device)

    test_logits = clf.logits(test_texts)
    pred = test_logits.argmax(1)
    out = {
        "name": args.name,
        "task": args.task,
        "text_col": args.text_col,
        "model": args.model_dir.name,
        "parameters": int(sum(p.numel() for p in clf.model.parameters())),
        "labels": list(labels),
        "train_size": len(ids),
        "valid_size": len(valid),
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "batch": args.batch,
            "max_len": args.max_len,
            "warmup_steps": warmup,
            "weight_decay": 0.01,
            "class_weight": args.class_weight,
        },  # fmt: skip
        "device": args.device,
        "history": history,
        "best_epoch": best[1],
        "best_valid_macro_f1": round(best[0], 4),
        "train_seconds": round(train_seconds, 1),
        "train_seconds_to_best_epoch": round(sum(h["train_seconds"] for h in history[: best[1]]), 1),
        "validation_seconds": round(eval_seconds, 1),
        "steps": step,
        "test": C.summary(y_test, pred, labels),
        "environment": C.environment(),
    }
    print(
        f"{args.name}: best epoch {best[1]} | test acc={out['test']['accuracy']:.4f} macro_f1={out['test']['macro_f1']:.4f}",
        flush=True,
    )
    np.savez_compressed(
        run_dir / "predictions.npz", y_test=y_test.astype(np.int8), test_logits=test_logits.astype(np.float16)
    )

    if not args.no_battery:
        train_cleaned = set(frames["train"]["text_cleaned"].astype(str))
        out["bootstrap"] = C.bootstrap(y_test, pred, k)
        out["subgroups"] = C.subgroups(test, pred, labels, train_cleaned)
        out["privacy"] = C.privacy_probe(clf.predict, C.stratified_sample(test, args.probe)["x"].tolist())
        if args.task == "p3":
            out["suite"] = C.run_suite(lambda texts: [labels[i] for i in clf.predict(texts)])
            print(
                f"  suite {out['suite']['passed']}/{out['suite']['n']} | privacy {out['privacy']['label_changes']}",
                flush=True,
            )
        out[f"throughput_{args.device}"] = C.throughput(clf.predict, test_texts)
        if args.device != "cpu":
            clf.to("cpu")
            out["throughput_cpu"] = C.throughput(clf.predict, test_texts[:2000])
            out["throughput_cpu"]["threads"] = torch.get_num_threads()
            clf.to(args.device)
    if args.save_model:
        clf.model.save_pretrained(str(run_dir / "model"), safe_serialization=True)
        clf.tokenizer.save_pretrained(str(run_dir / "model"))
        out["model_megabytes"] = round(sum(f.stat().st_size for f in (run_dir / "model").glob("*")) / 1e6, 1)
    C.write_json(run_dir / "metrics.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
