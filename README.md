# CRYPTO ANALYSIS AND RISK ANALYZER

This project includes:
- A **Streamlit app** (`streamlit_app.py`) with login + live crypto data dashboard
- The original **Flask app** (`app.py`) with HTML templates

## Run with Streamlit (recommended)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Login:
- Username: `admin@gmail.com`
- Password: `123456`

## Run with Flask (optional)

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/login`

## Notes
- `crypto_data.csv` is generated automatically and is **not committed** to git.
- For Flask sessions, you can optionally set:

```bash
set FLASK_SECRET_KEY=your-secret-key
```

