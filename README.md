# Crypto Analysis and Risk Analyzer

**One project** — login, live crypto data (CoinGecko), realtime statistics, and user-addable data.  
Run with a single command: **`streamlit run app.py`**

---

## Run locally (one project, one command)

### 1. Clone the repo

```bash
git clone https://github.com/CHARITHA-14/CRYPTO-ANALYSIS-AND-RISK-ANALYZER.git
cd CRYPTO-ANALYSIS-AND-RISK-ANALYZER
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Open in browser

- App opens at **http://localhost:8501** (or the URL shown in the terminal).
- **Login:** `admin@gmail.com` / `123456`

---

## Deploy online (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and deploy.
3. Set **Main file path** to **`app.py`**.
4. The app runs at the URL Streamlit gives you.

Repo must contain **`app.py`** and **`requirements.txt`** (streamlit, requests, pandas).

---

## Features

- **Login** – Sign in with username/password.
- **Dashboard** – Live crypto table (top 5 from CoinGecko) plus your own entries.
- **Realtime statistics** – Last updated, total volume, avg 24h %, top gainer, top loser.
- **Add your data** – Form: name, symbol, price, 24h %, volume. Stored in `user_added_data.json`.

## Notes

- `user_added_data.json` holds user-added entries (in `.gitignore`).
- This is a single Streamlit app; there is no separate Flask project in this repo.
