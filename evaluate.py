"""Evaluation script (Review 3 - Performance Evaluation & Metrics).

Runs the full pipeline on random images from the Hugging Face dataset and reports:
  * Recognition accuracy and macro F1 (vs. the dataset's true labels)
  * Calibration: accuracy for each confidence level
  * ROUGE-L between the generated meaning and the verified reference meaning
  * Safety compliance (% of explanations with no live-driving commands)
  * Uncertainty test on degraded (blurred + darkened) images
  * Label-mismatch detection rate
  * Average response time

Examples:
  python evaluate.py --n 30                 # quick run
  python evaluate.py --n 100 --sleep 4      # bigger run, pause between calls (free-tier limits)
  python evaluate.py --n 30 --no-llm        # test recognition only (saves Gemini quota)
"""
import argparse
import random
import time

import pandas as pd
from PIL import ImageEnhance, ImageFilter

from src import config, dataset, pipeline, reference, safety


def canon(name: str) -> str:
    r = reference.lookup(name)
    if not r.found:
        return reference.normalise(name) or "unknown"
    v = reference.speed_value(name)
    return f"{r.name} {v}" if v and r.name == "Speed Limit" else r.name


def degrade(img):
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return ImageEnhance.Brightness(img).enhance(0.35)


def rouge_l(a: str, b: str) -> float:
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(a, b)["rougeL"].fmeasure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of dataset images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default=config.VLM_BACKEND, choices=["hf", "ollama"])
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between images")
    ap.add_argument("--no-llm", action="store_true", help="skip Gemini (use reference text)")
    ap.add_argument("--degraded", type=int, default=10, help="images for the blur/dark uncertainty test")
    ap.add_argument("--mismatch", type=int, default=10, help="images for the wrong-label test")
    args = ap.parse_args()

    def no_llm(system, user):
        raise RuntimeError("LLM disabled for this run")
    llm = no_llm if args.no_llm else None

    config.RESULTS_DIR.mkdir(exist_ok=True)
    samples = dataset.sample(args.n, seed=args.seed)
    all_labels = sorted({lab for _, lab in samples})
    rows = []

    print(f"Running {len(samples)} clean images...")
    for i, (img, true) in enumerate(samples, 1):
        res = pipeline.run(img, backend=args.backend, llm=llm)
        pred = res.sign_name if res.status in ("explained", "uncertain") else res.status
        ref_true = reference.lookup(true)
        text = " ".join(res.sections.values()) if res.sections else ""
        rows.append({
            "test": "clean", "true_label": true, "predicted": pred,
            "correct": pipeline.same_sign(true, pred), "status": res.status, "confidence": res.confidence,
            "explanation_source": res.explanation_source,
            "rougeL_meaning": round(rouge_l(ref_true.entry["meaning"], res.sections["meaning"]), 3)
            if (res.sections and ref_true.found and res.explanation_source == "gemini") else None,
            "safety_violations": "; ".join(safety.command_violations(text)),
            "seconds": round(res.seconds, 2), "error": res.message if res.status == "error" else "",
        })
        print(f"  {i}/{len(samples)} true={true!r} pred={pred!r} conf={res.confidence} {res.status}")
        time.sleep(args.sleep)

    rng = random.Random(args.seed)
    print(f"Running {args.degraded} degraded images...")
    for img, true in samples[: args.degraded]:
        res = pipeline.run(degrade(img), backend=args.backend, llm=llm)
        rows.append({"test": "degraded", "true_label": true, "predicted": res.sign_name, "status": res.status,
                     "confidence": res.confidence, "correct": pipeline.same_sign(true, res.sign_name),
                     "seconds": round(res.seconds, 2)})
        time.sleep(args.sleep)

    print(f"Running {args.mismatch} wrong-label tests...")
    for img, true in samples[: args.mismatch]:
        wrong = rng.choice([x for x in reference.all_sign_names() if not pipeline.same_sign(x, true)])
        res = pipeline.run(img, user_label=wrong, backend=args.backend, llm=llm)
        rows.append({"test": "mismatch", "true_label": true, "given_label": wrong, "predicted": res.sign_name,
                     "status": res.status, "confidence": res.confidence, "label_match": res.label_match,
                     "seconds": round(res.seconds, 2)})
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "results.csv", index=False)

    # ---------- Metrics ----------
    from sklearn.metrics import f1_score
    clean = df[df.test == "clean"]
    valid = clean[clean.status != "error"]
    acc = valid.correct.mean() if len(valid) else 0
    f1 = f1_score([canon(t) for t in valid.true_label], [canon(p) for p in valid.predicted],
                  average="macro", zero_division=0) if len(valid) else 0
    calib = valid.groupby("confidence").correct.agg(["count", "mean"]).rename(columns={"mean": "accuracy"})
    explained = clean[clean.status == "explained"]
    safe = (explained.safety_violations.fillna("") == "").mean() if len(explained) else 1.0
    rouge = clean.rougeL_meaning.dropna().mean() if "rougeL_meaning" in clean else None
    deg = df[df.test == "degraded"]
    deg_flag = (deg.confidence != "high").mean() if len(deg) else None
    mm = df[df.test == "mismatch"]
    mm_det = (mm.label_match == "Mismatch").mean() if len(mm) else None

    lines = [
        "# Evaluation Summary - PS 113 Traffic Sign Explainer", "",
        f"- Dataset: `{config.HF_DATASET}` (random sample, seed={args.seed})",
        f"- Vision backend: {args.backend} ({config.QWEN_MODEL if args.backend == 'hf' else config.OLLAMA_MODEL})",
        f"- LLM: {'disabled' if args.no_llm else config.GEMINI_MODEL}", "",
        "| Metric | Value |", "|---|---|",
        f"| Images evaluated (clean) | {len(clean)} ({len(clean) - len(valid)} errors) |",
        f"| Recognition accuracy | {acc:.1%} |",
        f"| Macro F1 | {f1:.3f} |",
        f"| Explained (medium/high confidence) | {len(explained)} / {len(clean)} |",
        f"| Safety compliance (no driving commands) | {safe:.1%} |",
        f"| Mean ROUGE-L (meaning vs reference) | {rouge:.3f} |" if rouge == rouge and rouge is not None else "| Mean ROUGE-L | n/a (LLM disabled or no Gemini answers) |",
        f"| Degraded images flagged (not 'high') | {deg_flag:.1%} |" if deg_flag is not None else "",
        f"| Wrong-label detection rate | {mm_det:.1%} |" if mm_det is not None else "",
        f"| Average time per image (s) | {clean.seconds.mean():.2f} |", "",
        "## Calibration (accuracy by confidence level)", "", "\n".join(["| Confidence | Images | Accuracy |", "|---|---|---|"] + [f"| {c} | {int(r['count'])} | {r['accuracy']:.1%} |" for c, r in calib.iterrows()]) if len(calib) else "n/a", "",
        "Higher confidence should show higher accuracy. Full per-image results: `results/results.csv`.",
    ]
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
