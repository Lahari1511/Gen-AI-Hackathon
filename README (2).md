---
title: PS 113 Traffic Sign Explainer
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

# 🚦 Traffic Sign Educational Explanation Tool (PS 113)

Upload a traffic sign photo → the app identifies the sign with **Qwen2.5-VL** (Vision Language Model) → checks it against **verified meanings** (based on IRC:67-2022) → **Gemini** (LLM) writes a simple explanation in 4 parts: **Meaning, Purpose, Importance, Uncertainty**.

> ⚠️ Educational use only. The app never gives live driving instructions.

---

## 1. What you need
- **Python 3.11** (python.org) and **VS Code**
- A free **Gemini API key** → https://aistudio.google.com → *Get API key*
- A free **Hugging Face token** → https://huggingface.co/settings/tokens → *Create new token* (Read)

## 2. Setup in VS Code (one time)
1. Unzip this folder and open it in VS Code (**File → Open Folder**).
2. Open the terminal (**Terminal → New Terminal**).
3. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)
   source venv/bin/activate       (Mac/Linux)
   ```
4. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
5. Copy `.env.example` to a new file called `.env` and paste your two keys into it.

## 3a. Website version (easiest)
Open `website/index.html` in Chrome, add your keys in ⚙ Settings, upload a sign. See `website/README.md`.

## 3. Run the app
> To put it online as a website, follow **DEPLOY.md**.

```
streamlit run app.py
```
Your browser opens at `http://localhost:8501`.
- Upload a sign image **or** click **Load random dataset image** (sidebar).
- Optionally type the class label, then click **Explain this sign**.

## 4. Check it works (no internet needed)
```
python -m pytest -v
```
All 27 tests should pass (see `TEST_RESULTS.md`).

## 5. Explore the dataset (Review 2 – Data Preparation)
```
python explore_dataset.py
```
Prints rows, classes and class counts, and saves `results/dataset_classes.csv` and `results/dataset_samples.png`.
It also lists dataset classes that have **no verified meaning yet** → add them to `data/reference_signs.json`.

## 6. Evaluate (Review 3 – Metrics)
```
python evaluate.py --n 30 --sleep 4
```
Creates `results/summary.md` with accuracy, macro F1, calibration, ROUGE-L, safety compliance, blur test and wrong-label test.
Use `--no-llm` to test recognition only and save Gemini quota.

---

## How it works (modules)

| # | Module | File | What it does |
|---|---|---|---|
| 1 | Image upload / UI | `app.py` | Streamlit page: upload, results, warnings |
| 1b | Dataset | `src/dataset.py` | Loads the Hugging Face dataset, random samples |
| 2 | Preprocessing | `src/preprocess.py` | Checks file, size, blur, brightness; resizes |
| 3 | Recognition | `src/vision.py` | Qwen2.5-VL via Hugging Face API or Ollama → JSON |
| 4 | Verification | `src/reference.py` + `data/reference_signs.json` | Matches the sign to a verified meaning |
| 5 | Prompts | `src/prompts.py` | Prompt v1 (first try) and v2 (final, grounded) |
| 6 | Explanation | `src/explain.py` | Gemini → 4 sections, validated, retry, safe fallback |
| 7 | Safety | `src/safety.py` | Refuses driving questions, blocks command-style output |
| — | Pipeline | `src/pipeline.py` | Connects everything and computes confidence |

### How confidence is decided
Starts from the Vision model's confidence, then goes **down** if: the image is blurry/dark/small, the model reports problems, the sign is not in the verified list, or your label doesn't match. It is **never "High"** if the image has quality problems.
- **High / Medium** → Gemini explanation (grounded on the verified meaning)
- **Low** or **unknown sign** → *"Cannot confirm"* + possible signs. **No generation** → prevents hallucination.

### Error handling
| Problem | What happens |
|---|---|
| Corrupt / wrong file | Clear error message |
| Not a traffic sign | "No traffic sign detected" |
| Hugging Face / Ollama down | Clear error, app keeps running |
| Gemini error, quota (429) or messy answer | Retries once, then shows the verified reference text |
| Gemini writes a driving command | Blocked, safe text shown |
| Driving question ("I'm driving, can I turn?") | Refused before any AI call |

---

## If Hugging Face doesn't work for Qwen2.5-VL
Some free accounts can't run every model. Options:
1. **Ollama (local):** install from https://ollama.com, then run `ollama pull qwen2.5vl:3b`, set `VLM_BACKEND=ollama` in `.env` (or choose it in the sidebar).
2. **Label only:** choose *Label only* in the sidebar to demo the Gemini explanation using a typed label.

## Notes
- Free tiers have rate limits — don't click Explain repeatedly; use `--sleep` in evaluation.
- Free-tier prompts may be used by providers to improve products — don't upload private photos (faces, number plates).
- `data/reference_signs.json` contains curated educational meanings based on IRC:67-2022 categories. Review it as a team.

## Credits
- Dataset: `kannanwisen/Indian-Traffic-Sign-Classification` (Hugging Face)
- VLM: Qwen2.5-VL (Alibaba Qwen Team) · LLM: Google Gemini
- Sign categories: Indian Roads Congress, IRC:67-2022 Code of Practice for Road Signs
