# 🌐 Website version (no Python needed)

`index.html` is the complete website: upload a traffic sign image → it checks the image →
Qwen2.5-VL recognises the sign → verified IRC:67-2022 reference → confidence → Gemini explains it
in 4 sections (Meaning, Purpose, Importance, Uncertainty).

## Run it on your computer
1. Double-click `index.html` (opens in Chrome / Edge).
2. Click **⚙ Settings** → paste your **Hugging Face token** and **Gemini API key** → **Save**.
3. Upload a sign image (or click a sample sign) → **Explain this sign**.

## Put it online (free link in 2 minutes)
1. Go to **https://app.netlify.com/drop** (sign in free).
2. Drag this whole `website` folder onto the page.
3. You get a link like `https://your-name.netlify.app` – share it.
   Each person enters their own keys in ⚙ Settings (keys stay only in their browser).

## Notes
- Keys are stored only in your browser (localStorage) and sent only to Hugging Face and Google.
- If Qwen2.5-VL gives a Hugging Face error, try model `Qwen/Qwen2.5-VL-32B-Instruct` in Settings.
- If Gemini is unavailable, the site still shows the verified reference explanation.
