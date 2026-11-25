# 🌐 Put the app online (free) – step by step

You get a public website link like `https://ps113-traffic-signs.streamlit.app`.
No coding needed – everything is done in the browser. Takes about 15 minutes.

---

## Option A – Streamlit Community Cloud (recommended)

### Step 1: Put the code on GitHub
1. Go to **https://github.com** and sign up / sign in (free).
2. Click **+** (top right) → **New repository**.
3. Name: `ps113-traffic-sign-tool` → choose **Public** → click **Create repository**.
4. On the next page click **"uploading an existing file"**.
5. Open the unzipped `traffic-sign-tool` folder on your computer, select **everything inside it**
   (app.py, src, data, tests, requirements.txt, README.md, .streamlit …) and drag it into the browser.
   - ⚠️ Do **NOT** upload your `.env` file (it has your secret keys).
6. Click **Commit changes**.

### Step 2: Deploy on Streamlit
1. Go to **https://share.streamlit.io** → **Continue with GitHub** → allow access.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `your-username/ps113-traffic-sign-tool`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose a name, e.g. `ps113-traffic-signs`
4. Click **Advanced settings**:
   - Python version: **3.11**
   - **Secrets** – paste (with your real keys):
     ```
     GEMINI_API_KEY = "your_gemini_key"
     HF_TOKEN = "your_huggingface_token"
     VLM_BACKEND = "hf"
     ```
5. Click **Deploy**. Wait 3–5 minutes while it installs.
6. 🎉 Your app is live. Copy the link and share it.

### If something goes wrong
| Message | Fix |
|---|---|
| `ModuleNotFoundError` | Check `requirements.txt` was uploaded |
| "Gemini API key ⚠️" in sidebar | Add/correct the key in **App settings → Secrets**, then **Reboot app** |
| Hugging Face error / model not available | In the sidebar choose **Label only** to demo Gemini, or try `QWEN_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"` in Secrets |
| App sleeping | Free apps sleep when unused – open the link 5 minutes before the demo and click **Wake up** |

> Ollama does **not** work on the cloud (it needs your own computer). Use the Hugging Face option online.

---

## Option B – Hugging Face Spaces
1. Go to **https://huggingface.co/new-space**.
2. Space name: `ps113-traffic-sign-tool` · SDK: **Streamlit** · Hardware: **CPU basic (free)** · Public → **Create Space**.
3. **Files → Add file → Upload files** → drag all project files (not `.env`) → **Commit**.
4. **Settings → Variables and secrets → New secret**: add `GEMINI_API_KEY` and `HF_TOKEN`.
5. The Space builds automatically. Link: `https://huggingface.co/spaces/your-username/ps113-traffic-sign-tool`

---

## Before the demo – checklist
- [ ] Open the live link 5 minutes early (wake it up)
- [ ] Test one clear sign, one blurry sign and the "I'm driving…" question
- [ ] Keep the local VS Code version ready as a backup (`streamlit run app.py`)
- [ ] Keep screenshots of good results in case the internet fails
