"""
Crypto Analysis and Risk Analyzer — single project.
Run: streamlit run app.py
"""
import json
import random
import os
import hashlib
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np
import streamlit as st

try:
    import altair as alt
except ImportError:
    alt = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
except ImportError:
    go = None
    px = None
    pio = None

# ---------------- CONFIG & DATA ----------------
USERNAME = os.getenv("APP_USERNAME", "admin@gmail.com")
PASSWORD = os.getenv("APP_PASSWORD", "123456")
BASE_DIR = Path(__file__).resolve().parent
USER_DATA_FILE = BASE_DIR / "user_added_data.json"
HISTORY_FILE = BASE_DIR / "history.csv"
ACCOUNTS_FILE = BASE_DIR / "user_accounts.json"


def _init_altair():
    if alt is None:
        return
    try:
        alt.data_transformers.disable_max_rows()
    except (AttributeError, Exception):
        pass


def load_user_data():
    if not USER_DATA_FILE.exists():
        return []
    try:
        with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_user_data(entries):
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

def _hash_pw(pw):
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def load_accounts():
    if not ACCOUNTS_FILE.exists():
        return []
    try:
        with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_account(email, pw_hash):
    accounts = load_accounts()
    if any(a.get("email") == email for a in accounts):
        return False
    accounts.append({"email": email, "pw": pw_hash})
    with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=2)
    return True


def fetch_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 5,
        "page": 1,
        "sparkline": True,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    crypto_list = []
    for coin in data:
        crypto_list.append({
            "id": coin.get("id", ""),
            "name": coin["name"],
            "symbol": coin["symbol"].upper(),
            "price": coin["current_price"],
            "change": coin["price_change_percentage_24h"] if coin.get("price_change_percentage_24h") is not None else 0,
            "volume": coin["total_volume"],
            "spark": (coin.get("sparkline_in_7d") or {}).get("price"),
            "source": "api",
        })
    return crypto_list

def _get_cmc_key():
    k = None
    try:
        k = st.secrets.get("CMC_API_KEY", None)
    except Exception:
        k = None
    if not k:
        k = os.getenv("CMC_API_KEY")
    if not k:
        k = st.session_state.get("cmc_key")
    return k

def fetch_cmc_listings(limit=6):
    key = _get_cmc_key()
    if not key:
        return []
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"convert": "USD", "sort": "market_cap", "limit": limit}
    headers = {"X-CMC_PRO_API_KEY": key}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    js = r.json()
    data = []
    for item in js.get("data", []):
        q = (item.get("quote") or {}).get("USD") or {}
        data.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "symbol": (item.get("symbol") or "").upper(),
                "price": q.get("price"),
                "change": q.get("percent_change_24h"),
                "volume": q.get("volume_24h"),
                "source": "cmc",
            }
        )
    return data

def fetch_cmc_ohlcv_history(cmc_id, start_iso, end_iso):
    key = _get_cmc_key()
    if not key:
        return pd.DataFrame()
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"
    params = {"id": str(cmc_id), "convert": "USD", "time_start": start_iso, "time_end": end_iso}
    headers = {"X-CMC_PRO_API_KEY": key}
    r = requests.get(url, params=params, headers=headers, timeout=25)
    r.raise_for_status()
    js = r.json()
    quotes = (((js.get("data") or {}).get("quotes")) or [])
    rows = []
    for q in quotes:
        t = pd.to_datetime(q.get("time_open"))
        c = (((q.get("quote") or {}).get("USD")) or {}).get("close")
        v = (((q.get("quote") or {}).get("USD")) or {}).get("volume")
        rows.append({"time": t, "close": c, "volume": v})
    return pd.DataFrame(rows).dropna()


def compute_risk_metrics(prices_df, benchmark_returns, risk_free_rate=0.0):
    """
    Compute Volatility (annualized), Sharpe ratio, Beta vs benchmark, and VaR (95%).
    prices_df: DataFrame with 'time' and 'close' (or 'price') columns
    benchmark_returns: Series of daily log returns for benchmark (e.g., BTC)
    """
    if prices_df.empty or len(prices_df) < 2:
        return None
    col = "close" if "close" in prices_df.columns else "price"
    prices = prices_df[col].dropna()
    if len(prices) < 2:
        return None
    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        return None
    # Align with benchmark
    bench = benchmark_returns.reindex(log_returns.index).dropna()
    asset = log_returns.reindex(bench.index).dropna()
    common_idx = asset.index.intersection(bench.index)
    if len(common_idx) < 2:
        return None
    asset = asset.loc[common_idx]
    bench = bench.loc[common_idx]
    # Daily and annualized volatility (252 trading days)
    daily_vol = float(asset.std())
    ann_vol = daily_vol * np.sqrt(252) * 100  # as percentage
    # Sharpe ratio (annualized, risk-free=0)
    mean_daily = float(asset.mean())
    sharpe = (mean_daily - risk_free_rate / 252) / daily_vol * np.sqrt(252) if daily_vol > 0 else 0.0
    # Beta vs benchmark
    cov = np.cov(asset, bench)[0, 1]
    var_bench = np.var(bench)
    beta = cov / var_bench if var_bench > 0 else 0.0
    # VaR 95% (parametric: mean - 1.645 * std)
    var_95 = float(np.percentile(asset, 5)) * 100  # as percentage
    return {
        "volatility": round(ann_vol, 2),
        "sharpe": round(sharpe, 2),
        "beta": round(beta, 2),
        "var": round(abs(var_95), 2),
    }


def fetch_price_history(coin_id, days=30):
    if not coin_id:
        return pd.DataFrame()
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days), "interval": "hourly"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    prices = js.get("prices", [])
    volumes = js.get("total_volumes", [])
    if not prices:
        return pd.DataFrame()
    vol_map = {}
    for v in volumes or []:
        vol_map[datetime.fromtimestamp(v[0] / 1000.0)] = float(v[1])
    rows = []
    for p in prices:
        t = datetime.fromtimestamp(p[0] / 1000.0)
        rows.append({"time": t, "price": float(p[1]), "volume": vol_map.get(t, None)})
    return pd.DataFrame(rows).set_index("time").sort_index()


def get_milestone2_data(days=30):
    """
    Fetch historical data for 6+ cryptos from CoinMarketCap (or CoinGecko fallback),
    compute risk metrics (Volatility, Sharpe, Beta, VaR).
    """
    target_cryptos = ["BTC", "ETH", "SOL", "ADA", "DOGE", "XRP"]
    cmc_to_gecko = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "ADA": "cardano", "DOGE": "dogecoin", "XRP": "ripple",
    }
    end = datetime.now()
    start = end - pd.Timedelta(days=days)
    key = _get_cmc_key()
    listings = []
    try:
        listings = fetch_cmc_listings(limit=10)
    except Exception:
        listings = []
    id_map = {}
    for d in listings:
        sym = (d.get("symbol") or "").upper()
        if sym in target_cryptos and d.get("id"):
            id_map[sym] = d
    for sym in target_cryptos:
        if sym not in id_map:
            id_map[sym] = {"symbol": sym, "name": sym, "id": None}
    frames = []
    for sym in target_cryptos:
        df = pd.DataFrame()
        info = id_map.get(sym, {})
        cmc_id = info.get("id")
        if key and cmc_id:
            try:
                df = fetch_cmc_ohlcv_history(cmc_id, start.isoformat(), end.isoformat())
            except Exception:
                df = pd.DataFrame()
        if df.empty and sym in cmc_to_gecko:
            try:
                gdf = fetch_price_history(cmc_to_gecko[sym], days=days)
                if not gdf.empty:
                    gdf = gdf.reset_index()
                    gdf = gdf.rename(columns={"price": "close"})
                    df = gdf[["time", "close"]].copy()
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            syn = _synthetic_history([sym], min(720, days * 24))
            syn = syn[syn["symbol"] == sym][["time", "price"]].rename(columns={"price": "close"})
            df = syn
        if not df.empty:
            df["symbol"] = sym
            df["name"] = info.get("name", sym)
            if "close" not in df.columns and "price" in df.columns:
                df["close"] = df["price"]
            frames.append(df)
    if not frames:
        return pd.DataFrame(), []
    all_df = pd.concat(frames, ignore_index=True)
    price_col = "close" if "close" in all_df.columns else "price"
    pivot = all_df.pivot_table(index="time", columns="symbol", values=price_col)
    pivot = pivot.sort_index().ffill().bfill()
    returns = np.log(pivot / pivot.shift(1)).dropna(how="all")
    if returns.empty or "BTC" not in returns.columns:
        return all_df, []
    bench = returns["BTC"].dropna()
    risk_free = 0.0
    metrics_list = []
    for sym in target_cryptos:
        if sym not in returns.columns:
            continue
        asset_ret = returns[sym].dropna()
        common_idx = asset_ret.index.intersection(bench.index)
        if len(common_idx) < 2:
            continue
        asset_ret = asset_ret.loc[common_idx]
        bench_ret = bench.loc[common_idx]
        daily_vol = float(asset_ret.std())
        ann_vol = daily_vol * np.sqrt(252) * 100 if daily_vol > 0 else 0.0
        mean_daily = float(asset_ret.mean())
        sharpe = (mean_daily - risk_free / 252) / daily_vol * np.sqrt(252) if daily_vol > 0 else 0.0
        cov = np.cov(asset_ret, bench_ret)[0, 1]
        var_bench = np.var(bench_ret)
        beta = cov / var_bench if var_bench > 0 else 0.0
        var_95 = float(np.percentile(asset_ret, 5)) * 100
        name = id_map.get(sym, {}).get("name") or sym
        metrics_list.append({
            "crypto": name,
            "symbol": sym,
            "volatility": round(ann_vol, 2),
            "sharpe": round(sharpe, 2),
            "beta": round(beta, 2),
            "var": round(abs(var_95), 2),
        })
    return all_df, metrics_list


def get_combined_data():
    data = []
    try:
        data = fetch_crypto_data()
    except (requests.RequestException, ValueError):
        data = []
    for u in load_user_data():
        data.append({
            "name": u.get("name", ""),
            "symbol": u.get("symbol", "").upper(),
            "price": float(u.get("price", 0)),
            "change": float(u.get("change", 0)),
            "volume": float(u.get("volume", 0)),
            "source": "user",
        })
    if not data:
        data = _placeholder_live_data()
    log_history(data)
    return data


def compute_stats(crypto_list):
    if not crypto_list:
        return {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_volume": 0,
            "avg_change": 0,
            "top_gainer": None,
            "top_loser": None,
            "count": 0,
        }
    df = pd.DataFrame(crypto_list)
    total_volume = df["volume"].sum()
    changes = df["change"].replace([None], 0).fillna(0)
    avg_change = float(changes.mean()) if len(changes) else 0
    sorted_up = df.sort_values("change", ascending=False)
    sorted_down = df.sort_values("change", ascending=True)
    top_gainer = sorted_up.iloc[0].to_dict() if len(sorted_up) else None
    top_loser = sorted_down.iloc[0].to_dict() if len(sorted_down) else None
    return {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_volume": int(total_volume),
        "avg_change": round(avg_change, 2),
        "top_gainer": top_gainer,
        "top_loser": top_loser,
        "count": len(crypto_list),
    }

def _placeholder_live_data():
    seed = int(datetime.now().timestamp()) + int(st.session_state.get("refresh_count", 0))
    rnd = random.Random(seed)
    assets = [
        ("Bitcoin", "BTC", 60000.0, (1.5e10, 4.0e10)),
        ("Ethereum", "ETH", 2800.0, (5.0e9, 2.0e10)),
        ("Solana", "SOL", 110.0, (8.0e8, 6.0e9)),
        ("XRP", "XRP", 0.55, (5.0e8, 3.0e9)),
        ("Cardano", "ADA", 0.45, (4.0e8, 2.5e9)),
    ]
    out = []
    for name, sym, base, (vmin, vmax) in assets:
        drift = rnd.uniform(-0.5, 0.5)
        vol_factor = rnd.uniform(0.5, 1.0)
        change = max(-12.0, min(12.0, rnd.gauss(drift, 3.0)))
        price = max(0.01, base * (1 + change / 100) * (1 + rnd.uniform(-0.0075, 0.0075)))
        vol_base = rnd.uniform(vmin, vmax)
        vol = vol_base * (1 + abs(change) / 40) * vol_factor
        out.append(
            {
                "id": sym.lower(),
                "name": name,
                "symbol": sym,
                "price": round(price, 4),
                "change": round(change, 2),
                "volume": int(vol),
                "spark": None,
                "source": "demo",
            }
        )
    return out

def _synthetic_history(symbols=None, points=240):
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL", "XRP", "ADA"]
    bases = {"BTC": 60000.0, "ETH": 2800.0, "SOL": 110.0, "XRP": 0.55, "ADA": 0.45}
    names = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "XRP": "XRP", "ADA": "Cardano"}
    end = datetime.now()
    times = pd.date_range(end=end, periods=points, freq="h")
    rows = []
    seed = int(datetime.now().timestamp())
    rnd = random.Random(seed)
    for sym in symbols:
        base = bases.get(sym, 100.0)
        price = base
        for t in times:
            step = rnd.uniform(-0.02, 0.02)
            price = max(0.01, price * (1 + step))
            change = step * 100
            vol = rnd.uniform(1e6, 1e9)
            rows.append(
                {
                    "time": t,
                    "name": names.get(sym, sym),
                    "symbol": sym,
                    "price": price,
                    "change": change,
                    "volume": vol,
                    "source": "demo",
                }
            )
    return pd.DataFrame(rows)


def log_history(entries):
    """Append current snapshot to a CSV so we can chart over time."""
    if not entries:
        return
    rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for e in entries:
        rows.append(
            {
                "time": ts,
                "name": e.get("name", ""),
                "symbol": e.get("symbol", ""),
                "price": e.get("price", 0),
                "change": e.get("change", 0),
                "volume": e.get("volume", 0),
                "source": e.get("source", ""),
            }
        )
    df = pd.DataFrame(rows)
    # write header only once
    write_header = not HISTORY_FILE.exists()
    df.to_csv(HISTORY_FILE, mode="a", header=write_header, index=False)


# ---------------- STREAMLIT UI ----------------
st.set_page_config(
    page_title="Crypto Analysis and Risk Analyzer",
    page_icon="💹",
    layout="wide",
)

_init_altair()


def _inject_custom_css(filename):
    css_path = BASE_DIR / "static" / filename
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        # Ensure CSS targets the Streamlit app container
        css = css.replace("body::before", ".stApp::before").replace("body > *", ".stApp > *")
        css = css.replace("body{", ".stApp{").replace("body {", ".stApp {")
        css += "\n.stApp { min-height: 100vh; }\n"
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _inject_fit_screen_css():
    """Injects CSS to make components fit the viewport better."""
    fit_css = """
    <style>
    .stApp { padding: 1rem 1.5rem 2rem; }
    [data-testid="stVerticalBlock"] > div { max-width: 100%; }
    .stDataFrame { max-height: 280px; overflow: auto; }
    [data-testid="stMetric"] { padding: 0.5rem; }
    .stColumns > div { min-width: 0; }
    @media (max-height: 800px) {
        .stApp { padding: 0.75rem 1rem 1.5rem; }
    }
    </style>
    """
    st.markdown(fit_css, unsafe_allow_html=True)


def _show_welcome():
    _inject_custom_css("welcome.css")
    st.session_state.setdefault("page", "welcome")
    st.markdown(
        """
        <div class="hero">
          <h1 class="title hero-title">Crypto Analysis & Risk Analyzer</h1>
          <p class="subtitle">Visualize volatility, track trends, and manage exposure.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Centered single row of Login/Signup buttons
    col_spacer1, col_btn1, col_btn2, col_spacer2 = st.columns([2, 1, 1, 2])
    with col_btn1:
        if st.button("Login", key="welcome_login", use_container_width=True):
            st.session_state["page"] = "login"
            st.rerun()
    with col_btn2:
        if st.button("Sign up", key="welcome_signup", use_container_width=True):
            st.session_state["page"] = "signup"
            st.rerun()
    st.markdown(
        """
        <div class="features">
          <div class="card">
            <div class="card-title">Live Prices</div>
            <div class="card-desc">Fetch top assets and visualize short-term trends.</div>
          </div>
          <div class="card">
            <div class="card-title">Volatility</div>
            <div class="card-desc">Spot peaks/lows with rolling bands and markers.</div>
          </div>
          <div class="card">
            <div class="card-title">Risk Controls</div>
            <div class="card-desc">Monitor volume, diversify, and set position limits.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_login():
    _inject_custom_css("login.css")
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("login_error", "")
    st.session_state.setdefault("page", "login")
    col_ill, col_form = st.columns([1.1, 1])
    with col_ill:
        img_path = BASE_DIR / "static" / "login-graphic.svg"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
    with col_form:
        st.markdown(
            "<p style='margin:0 0 6px 0;font-size:0.75rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#9ca3af'>"
            "Crypto Volatility &amp; Risk Analyzer</p>"
            "<h2 style='margin:0 0 4px 0;font-size:1.6rem;font-weight:600;color:#e5e7eb'>Welcome back</h2>"
            "<p style='margin:0 0 18px 0;font-size:0.9rem;color:#9ca3af;'>Sign in to access your realtime crypto risk dashboard.</p>",
            unsafe_allow_html=True,
        )
        username = st.text_input("Username", key="su")
        password = st.text_input("Password", type="password", key="sp")
        if st.button("Sign in"):
            ok = False
            if username == USERNAME and password == PASSWORD:
                ok = True
            else:
                accs = load_accounts()
                pw_hash = _hash_pw(password)
                for a in accs:
                    if a.get("email") == username and a.get("pw") == pw_hash:
                        ok = True
                        break
            if ok:
                st.session_state["authenticated"] = True
                st.session_state["login_error"] = ""
                st.session_state["page"] = "intro"
                st.rerun()
            else:
                st.session_state["login_error"] = "Invalid credentials"
        if st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])

def _show_signup():
    _inject_custom_css("login.css")
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("page", "signup")
    col_ill, col_form = st.columns([1.1, 1])
    with col_ill:
        img_path = BASE_DIR / "static" / "login-graphic.svg"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
    with col_form:
        st.markdown(
            "<p style='margin:0 0 6px 0;font-size:0.75rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#9ca3af'>"
            "Crypto Analysis &amp; Risk Analyzer</p>"
            "<h2 style='margin:0 0 4px 0;font-size:1.6rem;font-weight:600;color:#e5e7eb'>Create your account</h2>"
            "<p style='margin:0 0 18px 0;font-size:0.9rem;color:#9ca3af;'>Sign up to access your realtime crypto risk dashboard.</p>",
            unsafe_allow_html=True,
        )
        email = st.text_input("Email", key="reg_email")
        pw1 = st.text_input("Password", type="password", key="reg_pw1")
        pw2 = st.text_input("Confirm Password", type="password", key="reg_pw2")
        if st.button("Sign up"):
            if not email or not pw1 or not pw2:
                st.error("All fields are required.")
            elif pw1 != pw2:
                st.error("Passwords do not match.")
            else:
                created = save_account(email, _hash_pw(pw1))
                if not created:
                    st.error("Account already exists.")
                else:
                    st.success("Account created.")
                    st.session_state["authenticated"] = True
                    st.session_state["page"] = "intro"
                    st.rerun()


def _render_sidebar(page_key):
    st.sidebar.title("Crypto Volatility and Risk Analyzer")
    st.sidebar.markdown(f"**{page_key}**")
    st.sidebar.markdown("---")
    
    pages = [
        "Intro", 
        "Data Acquisition and Setup", 
        "Data Processing and Calculation", 
        "Visualization and Dashboard Development", 
        "Risk Classification and Reporting"
    ]
    
    page_map = {
        "intro": "intro",
        "data acquisition and setup": "dashboard",
        "data processing and calculation": "milestone2",
        "visualization and dashboard development": "milestone3",
        "risk classification and reporting": "milestone4",
    }
    
    current_page = st.session_state.get("page", "intro")
    index = 0
    for i, p in enumerate(pages):
        if page_map.get(p.lower()) == current_page:
            index = i
            break

    choice = st.sidebar.radio("Navigate", pages, index=index, key=f"nav_{current_page}")
    
    target = page_map.get(choice.lower(), "intro")
    if target != current_page:
        st.session_state["page"] = target
        st.rerun()
        
    if st.sidebar.button("Logout", key=f"logout_{current_page}"):
        st.session_state["authenticated"] = False
        st.session_state["login_error"] = ""
        st.rerun()


def _show_milestone2_dashboard():
    """Risk Metrics Dashboard - Milestone 2: Volatility, Sharpe, Beta, VaR."""
    _inject_custom_css("dashboard.css")
    _inject_fit_screen_css()

    _render_sidebar("Data Processing and Calculation")

    st.title("Data Processing and Calculation")
    st.caption("Live Analysis · Data from CoinMarketCap API (6+ cryptocurrencies)")

    # Timeframe selector (30D, 90D, 1Y)
    days_choice = st.radio(
        "Timeframe",
        ["30D", "90D", "1Y"],
        index=0,
        horizontal=True,
        key="m2_timeframe",
    )
    days_map = {"30D": 30, "90D": 90, "1Y": 365}
    days = days_map.get(days_choice, 30)

    with st.spinner("Fetching data from CoinMarketCap and computing risk metrics..."):
        _, metrics_list = get_milestone2_data(days=days)

    if not metrics_list:
        st.warning("Unable to fetch data. Ensure CMC_API_KEY is set (or CoinGecko fallback will be used).")
        return

    # Metrics table
    st.subheader("Risk Metrics Table")
    mdf = pd.DataFrame(metrics_list)
    display_df = mdf[["crypto", "volatility", "sharpe", "beta", "var"]].rename(columns={
        "crypto": "Crypto",
        "volatility": "Volatility (%)",
        "sharpe": "Sharpe",
        "beta": "Beta",
        "var": "VaR (95%)",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Volatility comparison bar chart
    st.subheader("Crypto Volatility Comparison")
    if alt is not None and not mdf.empty:
        vol_chart = alt.Chart(mdf).mark_bar().encode(
            x=alt.X("crypto:N", title="Cryptocurrency", sort="-y"),
            y=alt.Y("volatility:Q", title="Volatility (%)"),
            color=alt.Color("volatility:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["crypto:N", alt.Tooltip("volatility:Q", format=".2f")],
        ).properties(height=300).configure_axis(
            grid=True, gridColor="#374151", labelColor="#e5e7eb", titleColor="#e5e7eb",
        ).configure_view(stroke=None)
        st.altair_chart(vol_chart, use_container_width=True)
    else:
        st.bar_chart(mdf.set_index("crypto")["volatility"])

    # Historic Data Visualization (moved from Milestone 1)
    st.markdown("---")
    st.subheader("Historic Data Visualization")
    st.caption("Visualize price trends from stored history data over time.")
    _render_historic_visualization()


def _show_milestone3_dashboard():
    """Milestone 3: Visualization Dashboard - Interactive charts with Plotly."""
    _inject_custom_css("dashboard.css")
    _inject_fit_screen_css()

    _render_sidebar("Visualization and Dashboard Development")

    # Main content area
    st.title("Visualization and Dashboard Development")
    st.caption("Crypto Risk Analytics Dashboard · Interactive Analysis")

    # Cryptocurrency selection and date range
    col_controls1, col_controls2 = st.columns([2, 1])
    with col_controls1:
        selected_cryptos = st.multiselect(
            "Select Cryptocurrencies",
            ["BTC", "ETH", "SOL", "ADA", "DOGE", "XRP"],
            default=["BTC", "ETH", "SOL", "ADA"],
            key="m3_crypto_select"
        )
    with col_controls2:
        default_end = datetime.now()
        default_start = default_end - pd.Timedelta(days=365)
        date_range = st.date_input(
            "Date Range",
            value=(default_start.date(), default_end.date()),
            key="m3_date_range"
        )

    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency.")
        return

    # Fetch data for selected cryptos and date range
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = default_start.date()
        end_date = default_end.date()

    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    if days <= 0:
        days = 30

    with st.spinner("Loading data and computing metrics..."):
        all_df, metrics_list = get_milestone2_data(days=min(days, 365))
        
        # Filter metrics for selected cryptos
        selected_metrics = [m for m in metrics_list if m.get("symbol") in selected_cryptos]
        
        if not selected_metrics:
            st.error("No data available for selected cryptocurrencies.")
            return

    # Price & Volatility Trends Chart (First - Full Width)
    st.subheader("Price & Volatility Trends")
    if go is not None and all_df is not None and not all_df.empty:
        # Filter data for selected cryptos and date range
        plot_df = all_df[all_df["symbol"].isin(selected_cryptos)].copy()
        plot_df["time"] = pd.to_datetime(plot_df["time"])
        plot_df = plot_df[(plot_df["time"].dt.date >= start_date) & (plot_df["time"].dt.date <= end_date)]
        
        if not plot_df.empty:
            # Calculate rolling volatility
            plot_df = plot_df.sort_values("time")
            plot_df["volatility"] = plot_df.groupby("symbol")["close"].transform(lambda x: x.pct_change().rolling(window=7, min_periods=1).std() * 100)
            
            fig_price_vol = go.Figure()
            
            # Add price line (left axis)
            for symbol in selected_cryptos:
                symbol_data = plot_df[plot_df["symbol"] == symbol].sort_values("time")
                if not symbol_data.empty:
                    fig_price_vol.add_trace(go.Scatter(
                        x=symbol_data["time"],
                        y=symbol_data["close"],
                        name=f"{symbol} Price",
                        yaxis="y",
                        line=dict(width=2),
                        mode="lines"
                    ))
            
            # Add volatility line (right axis) - average across selected cryptos
            if len(selected_cryptos) > 0:
                vol_data = plot_df.groupby("time")["volatility"].mean().reset_index()
                fig_price_vol.add_trace(go.Scatter(
                    x=vol_data["time"],
                    y=vol_data["volatility"],
                    name="Avg Volatility (%)",
                    yaxis="y2",
                    line=dict(color="#f59e0b", width=2, dash="dash"),
                    mode="lines"
                ))
            
                fig_price_vol.update_layout(
                    xaxis=dict(title="Time"),
                    yaxis=dict(title="Price (USD)", side="left"),
                    yaxis2=dict(title="Volatility (%)", side="right", overlaying="y"),
                    height=400,
                    hovermode="x unified",
                    legend=dict(
                        x=1.02,
                        y=1,
                        xanchor="left",
                        yanchor="top",
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(255,255,255,0.2)",
                        borderwidth=1
                    ),
                    margin=dict(r=150),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e5e7eb'),
                    xaxis_gridcolor='#374151',
                    yaxis_gridcolor='#374151'
                )
            st.plotly_chart(fig_price_vol, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("No data available for the selected date range.")
    else:
        st.info("Plotly not available. Using fallback visualization.")
        if all_df is not None and not all_df.empty:
            plot_df = all_df[all_df["symbol"].isin(selected_cryptos)]
            if not plot_df.empty:
                st.line_chart(plot_df.pivot_table(index="time", columns="symbol", values="close"))

    st.markdown("---")
    
    # Risk-Return Analysis Chart (Below - Full Width)
    st.subheader("Risk-Return Analysis")
    if go is not None and selected_metrics and all_df is not None and not all_df.empty:
        # Calculate actual returns from price data
        scatter_data = []
        plot_df_all = all_df[all_df["symbol"].isin(selected_cryptos)].copy()
        plot_df_all["time"] = pd.to_datetime(plot_df_all["time"])
        plot_df_all = plot_df_all.sort_values("time")
        
        for m in selected_metrics:
            symbol = m.get("symbol", "")
            symbol_data = plot_df_all[plot_df_all["symbol"] == symbol].sort_values("time")
            if len(symbol_data) > 1:
                # Calculate total return over the period
                initial_price = symbol_data["close"].iloc[0]
                final_price = symbol_data["close"].iloc[-1]
                total_return = ((final_price - initial_price) / initial_price) * 100 if initial_price > 0 else 0
            else:
                total_return = 0
            
            scatter_data.append({
                "symbol": symbol,
                "volatility": m.get("volatility", 0),
                "sharpe": m.get("sharpe", 0),
                "return_pct": total_return,
                "beta": m.get("beta", 0)
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        if not scatter_df.empty:
            # Map Sharpe ratio to positive size values (0-100 range)
            # Sharpe can be negative, so we normalize: abs(sharpe) scaled to 10-100
            min_sharpe = scatter_df["sharpe"].min()
            max_sharpe = scatter_df["sharpe"].max()
            sharpe_range = max_sharpe - min_sharpe if max_sharpe != min_sharpe else 1
            scatter_df["size_normalized"] = ((scatter_df["sharpe"] - min_sharpe) / sharpe_range * 90 + 10).clip(lower=10, upper=100)
            
            fig_scatter = px.scatter(
                scatter_df,
                x="volatility",
                y="return_pct",
                color="symbol",
                size="size_normalized",
                hover_data=["symbol", "volatility", "sharpe", "beta"],
                labels={
                    "volatility": "Volatility (%)",
                    "return_pct": "Return (%)",
                    "symbol": "Cryptocurrency",
                    "sharpe": "Sharpe Ratio",
                    "size_normalized": "Sharpe (normalized)"
                },
                color_discrete_map={
                    "BTC": "#f7931a", "ETH": "#627eea", "SOL": "#9945ff",
                    "ADA": "#0033ad", "DOGE": "#c2a633", "XRP": "#23292f"
                }
            )
            
            fig_scatter.update_layout(
                height=400,
                legend=dict(
                    x=1.02,
                    y=1,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(255,255,255,0.2)",
                    borderwidth=1
                ),
                margin=dict(r=150),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                xaxis_gridcolor='#374151',
                yaxis_gridcolor='#374151'
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("No data available for scatter plot.")
    else:
        st.info("Plotly not available or no metrics data.")

    # Key metrics row
    st.markdown("---")
    if selected_metrics:
        # Calculate aggregate metrics (average or weighted)
        avg_vol = sum(m.get("volatility", 0) for m in selected_metrics) / len(selected_metrics)
        avg_sharpe = sum(m.get("sharpe", 0) for m in selected_metrics) / len(selected_metrics)
        avg_beta = sum(m.get("beta", 0) for m in selected_metrics) / len(selected_metrics)
        
        # Determine risk level
        if avg_vol < 3:
            risk_level = "Low"
        elif avg_vol < 6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Volatility", f"{avg_vol:.2f}%")
        with metric_col2:
            st.metric("Sharpe Ratio", f"{avg_sharpe:.2f}")
        with metric_col3:
            st.metric("Beta vs BTC", f"{avg_beta:.2f}")
        with metric_col4:
            st.metric("Risk Level", risk_level)


def _show_milestone4_dashboard():
    _inject_custom_css("dashboard.css")
    _inject_fit_screen_css()

    _render_sidebar("Risk Classification and Reporting")

    st.title("Risk Classification and Reporting")
    st.caption("Define thresholds, highlight high-risk assets, and export reports.")

    days_opt = st.selectbox("Lookback Window", ["90 days", "180 days", "365 days"], index=2, key="m4_days")
    days_map = {"90 days": 90, "180 days": 180, "365 days": 365}
    days = days_map.get(days_opt, 365)

    c1, c2 = st.columns(2)
    with c1:
        low_med = st.slider("Low/Medium cutoff (volatility %)", min_value=1, max_value=10, value=3, key="m4_lm")
    with c2:
        med_high = st.slider("Medium/High cutoff (volatility %)", min_value=low_med+1, max_value=20, value=6, key="m4_mh")

    with st.spinner("Computing metrics and classifications..."):
        all_df, metrics_list = get_milestone2_data(days=days)
    if not metrics_list:
        st.info("No metrics available.")
        return
    mdf = pd.DataFrame(metrics_list)
    def classify_row(r):
        lvl = "Low"
        if r.get("volatility", 0) >= med_high or r.get("sharpe", 0) < 0:
            lvl = "High"
        elif r.get("volatility", 0) >= low_med:
            lvl = "Medium"
        return lvl
    mdf["risk_level"] = mdf.apply(classify_row, axis=1)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Assets", len(mdf))
    col_b.metric("Average Volatility", f"{mdf['volatility'].mean():.2f}%")
    high_cnt = int((mdf["risk_level"] == "High").sum())
    med_cnt = int((mdf["risk_level"] == "Medium").sum())
    low_cnt = int((mdf["risk_level"] == "Low").sum())
    col_c.metric("High Risk", high_cnt)
    col_d.metric("Medium Risk", med_cnt)

    st.markdown("---")
    st.subheader("Risk Classification")
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown('<div class="risk-card high"><div class="risk-title">High Risk</div></div>', unsafe_allow_html=True)
        for _, r in mdf[mdf["risk_level"] == "High"].sort_values("volatility", ascending=False).iterrows():
            st.markdown(f'<div class="risk-item"><span class="risk-badge high">HIGH</span> {r["crypto"]} · {r["volatility"]:.2f}%</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="risk-card medium"><div class="risk-title">Medium Risk</div></div>', unsafe_allow_html=True)
        for _, r in mdf[mdf["risk_level"] == "Medium"].sort_values("volatility", ascending=False).iterrows():
            st.markdown(f'<div class="risk-item"><span class="risk-badge medium">MED</span> {r["crypto"]} · {r["volatility"]:.2f}%</div>', unsafe_allow_html=True)
    with cc:
        st.markdown('<div class="risk-card low"><div class="risk-title">Low Risk</div></div>', unsafe_allow_html=True)
        for _, r in mdf[mdf["risk_level"] == "Low"].sort_values("volatility", ascending=False).iterrows():
            st.markdown(f'<div class="risk-item"><span class="risk-badge low">LOW</span> {r["crypto"]} · {r["volatility"]:.2f}%</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Risk Summary Report")
    left, right = st.columns([2, 1])
    with left:
        if px is not None:
            fig_pie = px.pie(
                values=[high_cnt, med_cnt, low_cnt],
                names=["High Risk", "Medium Risk", "Low Risk"],
                color=["High Risk", "Medium Risk", "Low Risk"],
                color_discrete_map={"High Risk": "#dc2626", "Medium Risk": "#f59e0b", "Low Risk": "#16a34a"},
                hole=0.35
            )
            fig_pie.update_layout(
                height=360,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb')
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": True})
        else:
            st.bar_chart(pd.Series({"High": high_cnt, "Medium": med_cnt, "Low": low_cnt}))
    with right:
        st.write("Totals")
        st.write(f"• Total Cryptocurrencies: {len(mdf)}")
        st.write(f"• Average Volatility: {mdf['volatility'].mean():.2f}%")
        st.write(f"• Risk Distribution: {high_cnt} High / {med_cnt} Medium / {low_cnt} Low")

    st.markdown("---")
    st.subheader("Exports")
    exp_df = mdf[["crypto", "symbol", "volatility", "sharpe", "beta", "var", "risk_level"]].copy()
    csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, file_name="risk_summary.csv", mime="text/csv", key="m4_csv")

    png_bytes = None
    if pio is not None and px is not None:
        try:
            fig_dist = px.bar(
                x=["High", "Medium", "Low"],
                y=[high_cnt, med_cnt, low_cnt],
                color=["High", "Medium", "Low"],
                color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"},
                title="Risk Distribution"
            )
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb')
            )
            png_bytes = pio.to_image(fig_dist, format="png", scale=2)
        except Exception:
            png_bytes = None
    if png_bytes:
        st.download_button("Download PNG", png_bytes, file_name="risk_summary.png", mime="image/png", key="m4_png")
    else:
        st.caption("PNG export requires 'kaleido'. Use the chart's modebar or install kaleido.")

    pdf_bytes = None
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from io import BytesIO
        bio = BytesIO()
        c = canvas.Canvas(bio, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, 750, "Crypto Risk Summary")
        c.setFont("Helvetica", 10)
        c.drawString(40, 730, f"Total: {len(mdf)}")
        c.drawString(40, 715, f"Average Volatility: {mdf['volatility'].mean():.2f}%")
        c.drawString(40, 700, f"Distribution: {high_cnt} High / {med_cnt} Medium / {low_cnt} Low")
        y = 670
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Crypto")
        c.drawString(200, y, "Volatility")
        c.drawString(280, y, "Sharpe")
        c.drawString(340, y, "Beta")
        c.drawString(390, y, "VaR")
        c.drawString(440, y, "Risk")
        c.setFont("Helvetica", 10)
        y -= 15
        for _, r in exp_df.sort_values(["risk_level", "volatility"], ascending=[False, False]).iterrows():
            c.drawString(40, y, str(r["crypto"])[:18])
            c.drawRightString(260, y, f"{r['volatility']:.2f}%")
            c.drawRightString(320, y, f"{r['sharpe']:.2f}")
            c.drawRightString(380, y, f"{r['beta']:.2f}")
            c.drawRightString(430, y, f"{r['var']:.2f}%")
            c.drawString(440, y, r["risk_level"])
            y -= 14
            if y < 60:
                c.showPage()
                y = 750
        c.save()
        pdf_bytes = bio.getvalue()
    except Exception:
        pdf_bytes = None
    if pdf_bytes:
        st.download_button("Download PDF", pdf_bytes, file_name="risk_summary.pdf", mime="application/pdf", key="m4_pdf")
    else:
        st.caption("PDF export requires 'reportlab'.")

def _show_dashboard():
    # optional dashboard-specific styling
    _inject_custom_css("dashboard.css")
    _inject_fit_screen_css()

    _render_sidebar("Data Acquisition and Setup")

    # Page title
    st.title("Data Acquisition and Setup")
    st.caption("Live crypto prices, CMC trends, and API data connectivity.")

    # Initialize refresh counter in session state
    st.session_state.setdefault("refresh_count", 0)
    
    data_all = get_combined_data()
    stats_data = compute_stats(data_all)

    # Realtime stat strip
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets tracked", stats_data["count"])
    c2.metric("Total volume", f"{stats_data['total_volume'] / 1e9:.2f}B" if stats_data["total_volume"] else "—")
    c3.metric("Avg 24h %", f"{stats_data['avg_change']}%")
    c4.metric("Last updated", stats_data["last_updated"][:16])

    g = stats_data.get("top_gainer")
    l = stats_data.get("top_loser")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("### Top gainer")
        if g:
            st.markdown(f"**{g.get('name', g.get('symbol', '—'))}**  \n{g.get('change', 0):.2f}% 24h")
        else:
            st.caption("No data")
    with c6:
        st.markdown("### Top loser")
        if l:
            st.markdown(f"**{l.get('name', l.get('symbol', '—'))}**  \n{l.get('change', 0):.2f}% 24h")
        else:
            st.caption("No data")

    # Header row with refresh button
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader("Crypto Data Fetcher · Live")
    with header_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key="refresh_data_btn", use_container_width=True):
            st.session_state["refresh_count"] += 1
            st.session_state["last_refresh"] = datetime.now().isoformat()
            st.rerun()
    if not data_all:
        data_all = _placeholder_live_data()
    df = pd.DataFrame(data_all)
    df["volume"] = df["volume"].apply(
        lambda x: f"{x/1e9:.2f}B" if x >= 1e9 else f"{x/1e6:.2f}M" if x >= 1e6 else str(x)
    )
    st.dataframe(
        df[["name", "symbol", "price", "change", "volume", "source"]].rename(
            columns={
                "name": "Name",
                "symbol": "Symbol",
                "price": "Price ($)",
                "change": "24h %",
                "volume": "Volume",
                "source": "Source",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### CMC Price Trends & Correlation")
    _render_cmc_trends_and_corr()

def _show_intro():
    _inject_custom_css("dashboard.css")
    _inject_fit_screen_css()

    _render_sidebar("Intro")

    st.title("Crypto Analysis & Risk Analyzer")
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
          <p style="margin:0;font-size:0.95rem;color:#94a3b8;">
            Welcome. This app covers four milestones: <strong>Data Acquisition and Setup</strong>, 
            <strong>Data Processing and Calculation</strong>, <strong>Visualization and Dashboard Development</strong>, and 
            <strong>Risk Classification and Reporting</strong>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📘 What is Cryptocurrency?")
    st.markdown(
        "Cryptocurrency is digital money that exists online and runs on blockchain technology, "
        "without control from banks or governments. Coins like Bitcoin and Ethereum can be used "
        "for trading, investing, and online transactions. Their prices change frequently based on "
        "demand, news, and market activity."
    )

    st.subheader("📊 Data Acquisition and Setup")
    st.markdown(
        "Fetch and store daily price data for 5+ cryptocurrencies (e.g. BTC, ETH, SOL, ADA, DOGE). "
        "View live prices, 24h change, volume, CMC price trends, and correlation. Data is stored locally (CSV) "
        "and API connectivity is verified."
    )

    st.subheader("📈 Data Processing and Calculation")
    st.markdown(
        "Compute risk metrics: daily returns, volatility (daily/annualized), Sharpe ratio, Beta, and VaR. "
        "View the risk metrics table, volatility comparison chart, and **Historic Data Visualization** "
        "with price line, candlestick-style, and volume+price charts over time."
    )

    st.subheader("📊 Visualization and Dashboard Development")
    st.markdown(
        "Interactive dashboard with Plotly visualizations: **Price & Volatility Trends** (dual-axis time-series), "
        "**Risk-Return Analysis** (scatter plot), multi-crypto selection, date range filtering, and aggregate "
        "risk metrics (Volatility, Sharpe Ratio, Beta vs BTC, Risk Level)."
    )

    st.subheader("📊 Risk Classification and Reporting")
    st.markdown(
        "Define thresholds for classifying assets into Low, Medium, and High risk; "
        "highlight high-risk assets; and export summary reports (CSV, PNG, PDF)."
    )

    st.markdown("---")
    
    # Dashboard Features section
    st.subheader("Dashboard Features")
    if go is not None:
        # Radar chart for dashboard features
        features = ['User Interface', 'Data Visualization', 'Risk Analysis', 'Performance', 'Interactivity']
        values = [95, 95, 70, 80, 90]  # Values matching the image description
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=features,
            fill='toself',
            name='Features',
            line_color='#3b82f6'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb', size=12)
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Plotly not available. Dashboard features radar chart cannot be displayed.")

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Go to Data Acquisition and Setup ➜", key="intro_m1", use_container_width=True):
            st.session_state["page"] = "dashboard"
            st.rerun()
    with col2:
        if st.button("Go to Data Processing and Calculation ➜", key="intro_m2", use_container_width=True):
            st.session_state["page"] = "milestone2"
            st.rerun()
    with col3:
        if st.button("Go to Visualization and Dashboard Development ➜", key="intro_m3", use_container_width=True):
            st.session_state["page"] = "milestone3"
            st.rerun()
    with col4:
        if st.button("Go to Risk Classification and Reporting ➜", key="intro_m4", use_container_width=True):
            st.session_state["page"] = "milestone4"
            st.rerun()

def _render_historic_visualization(key_prefix="m2_"):
    """Render historic data visualization. key_prefix avoids widget key clashes (e.g. m2_ for Milestone 2)."""
    if not HISTORY_FILE.exists():
        hist_df = _synthetic_history()
    else:
        try:
            hist_df = pd.read_csv(HISTORY_FILE, parse_dates=["time"])
            if hist_df.empty:
                hist_df = _synthetic_history()
        except Exception:
            hist_df = _synthetic_history()

    # Get unique symbols
    symbols = sorted(hist_df["symbol"].dropna().unique().tolist())
    if not symbols:
        hist_df = _synthetic_history()
        symbols = sorted(hist_df["symbol"].dropna().unique().tolist())

    # Controls for visualization (unique keys per context)
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_symbol = st.selectbox("Select cryptocurrency", symbols, key=f"{key_prefix}hist_symbol")
    with col2:
        time_range = st.selectbox(
            "Time range",
            ["All time", "Last 24 hours", "Last 7 days", "Last 30 days"],
            key=f"{key_prefix}hist_range"
        )
    with col3:
        chart_type = st.selectbox(
            "Chart type",
            ["Price Line", "Candlestick-style", "Volume + Price"],
            key=f"{key_prefix}hist_chart_type",
            index=0,
        )

    # Filter data
    symbol_data = hist_df[hist_df["symbol"] == selected_symbol].copy()
    symbol_data = symbol_data.sort_values("time")

    # Apply time range filter
    now = datetime.now()
    if time_range == "Last 24 hours":
        symbol_data = symbol_data[symbol_data["time"] >= now - pd.Timedelta(hours=24)]
    elif time_range == "Last 7 days":
        symbol_data = symbol_data[symbol_data["time"] >= now - pd.Timedelta(days=7)]
    elif time_range == "Last 30 days":
        symbol_data = symbol_data[symbol_data["time"] >= now - pd.Timedelta(days=30)]

    if symbol_data.empty:
        symbol_data = _synthetic_history([selected_symbol], 240)

    # Prepare data: ensure required columns and drop invalid rows for charts
    symbol_data["price"] = pd.to_numeric(symbol_data["price"], errors="coerce")
    symbol_data["volume"] = pd.to_numeric(symbol_data["volume"], errors="coerce")
    symbol_data["change"] = pd.to_numeric(symbol_data["change"], errors="coerce")
    if "name" not in symbol_data.columns:
        symbol_data["name"] = selected_symbol
    symbol_data = symbol_data.dropna(subset=["price"]).sort_values("time").reset_index(drop=True)
    if len(symbol_data) < 2:
        symbol_data = _synthetic_history([selected_symbol], max(48, len(symbol_data) * 2))
        symbol_data = symbol_data[symbol_data["symbol"] == selected_symbol].dropna(subset=["price"]).reset_index(drop=True)

    # Calculate statistics
    latest_price = symbol_data["price"].iloc[-1]
    earliest_price = symbol_data["price"].iloc[0]
    price_change = latest_price - earliest_price
    price_change_pct = (price_change / earliest_price * 100) if earliest_price != 0 else 0
    max_price = symbol_data["price"].max()
    min_price = symbol_data["price"].min()
    avg_price = symbol_data["price"].mean()

    # Display metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Latest Price", f"${latest_price:,.2f}")
    with m2:
        st.metric("Price Change", f"${price_change:,.2f}", f"{price_change_pct:.2f}%")
    with m3:
        st.metric("High", f"${max_price:,.2f}")
    with m4:
        st.metric("Low", f"${min_price:,.2f}")
    with m5:
        st.metric("Average", f"${avg_price:,.2f}")

    if chart_type == "Price Line":
        _render_price_line_chart(symbol_data, selected_symbol)
    elif chart_type == "Candlestick-style":
        _render_candlestick_chart(symbol_data, selected_symbol)
    else:
        _render_volume_price_chart(symbol_data, selected_symbol)

    # Show data table
    with st.expander("View Raw Data"):
        col_map = {"time": "Time", "name": "Name", "symbol": "Symbol", "price": "Price ($)", "change": "24h Change (%)", "volume": "Volume"}
        cols = [c for c in ["time", "name", "symbol", "price", "change", "volume"] if c in symbol_data.columns]
        display_df = symbol_data[cols].rename(columns=col_map)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button for filtered data
        csv_data = symbol_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {selected_symbol} History CSV",
            data=csv_data,
            file_name=f"{selected_symbol.lower()}_history.csv",
            mime="text/csv",
            key=f"{key_prefix}dl_hist_viz"
        )


def _render_price_line_chart(data, symbol):
    """Render a zoomable price line chart with directional color scheme"""
    data = data.copy()
    data["time"] = pd.to_datetime(data["time"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price"]).sort_values("time").reset_index(drop=True)
    if len(data) < 2:
        st.line_chart(data.set_index("time")["price"], height=340)
        return
    if alt is not None:
        data["price_change"] = data["price"].diff().fillna(0)
        data["rolling_mean"] = data["price"].rolling(window=min(5, len(data)), min_periods=1).mean()
        data["rolling_std"] = data["price"].rolling(window=min(5, len(data)), min_periods=1).std().fillna(0)
        data["upper_band"] = data["rolling_mean"] + (data["rolling_std"] * 2.5)
        data["lower_band"] = data["rolling_mean"] - (data["rolling_std"] * 2.5)
        data["is_peak"] = False
        data["is_valley"] = False
        for i in range(2, len(data) - 2):
            if data["price"].iloc[i] > max(data["price"].iloc[i-2:i]) and \
               data["price"].iloc[i] > max(data["price"].iloc[i+1:i+3]):
                data.loc[data.index[i], "is_peak"] = True
            if data["price"].iloc[i] < min(data["price"].iloc[i-2:i]) and \
               data["price"].iloc[i] < min(data["price"].iloc[i+1:i+3]):
                data.loc[data.index[i], "is_valley"] = True

        src = data.reset_index(drop=True)

        # Create interactive selection for zooming and panning
        brush = alt.selection_interval(bind='scales', encodings=['x'])
        
        # Format time axis properly
        time_axis = alt.X(
            "time:T", 
            title="Time",
            axis=alt.Axis(
                format="%b %d, %Y %H:%M",
                labelAngle=-45,
                tickCount=8
            )
        ).scale(domain=brush)
        
        # Bollinger bands with subtle color
        band = alt.Chart(src).mark_area(opacity=0.15, color="#64748b").encode(
            x=time_axis,
            y=alt.Y("lower_band:Q", title="Price ($)"),
            y2="upper_band:Q",
        )

        base = alt.Chart(src)
        price_line = base.mark_line(strokeWidth=3, interpolate='linear').encode(
            x=time_axis,
            y=alt.Y("price:Q", title="Price ($)"),
            color=alt.condition(
                "datum.price_change >= 0",
                alt.value("#22c55e"),
                alt.value("#ef4444")
            ),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %d, %Y %H:%M"),
                alt.Tooltip("price:Q", title="Price", format="$.2f"),
            ],
        )

        # Rolling mean line in amber
        mean_line = alt.Chart(src).mark_line(color="#f59e0b", strokeWidth=2, strokeDash=[4, 3]).encode(
            x=time_axis,
            y="rolling_mean:Q",
        )
        
        # Peaks and valleys
        peaks = src[src["is_peak"]]
        valleys = src[src["is_valley"]]
        
        peak_points = alt.Chart(peaks).mark_point(color="#ef4444", size=80, filled=True).encode(
            x=time_axis,
            y="price:Q",
            tooltip=[alt.Tooltip("time:T", title="Peak Time", format="%b %d, %Y %H:%M"), alt.Tooltip("price:Q", title="Peak Price", format="$.2f")],
        )
        
        valley_points = alt.Chart(valleys).mark_point(color="#22c55e", size=80, filled=True).encode(
            x=time_axis,
            y="price:Q",
            tooltip=[alt.Tooltip("time:T", title="Valley Time", format="%b %d, %Y %H:%M"), alt.Tooltip("price:Q", title="Valley Price", format="$.2f")],
        )

        chart = (band + price_line + mean_line + peak_points + valley_points).add_params(brush).properties(
            title=f"{symbol} Price History (Green=Up, Red=Down)",
            height=340
        ).configure_axis(
            grid=True,
            gridColor="#374151",
            labelColor="#e5e7eb",
            titleColor="#e5e7eb",
        ).configure_view(
            stroke=None
        ).configure_title(
            color="#e5e7eb"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(data.set_index("time")["price"], height=340)


def _render_candlestick_chart(data, symbol):
    """Render a candlestick-style chart using OHLC aggregation"""
    data = data.copy()
    data["time"] = pd.to_datetime(data["time"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price"]).sort_values("time")
    if len(data) < 2:
        st.line_chart(data.set_index("time")["price"], height=320)
        return
    data = data.set_index("time")

    # Determine resampling period from data timespan and density
    n_points = len(data)
    time_span_hours = (data.index.max() - data.index.min()).total_seconds() / 3600 if n_points > 1 else 24
    if time_span_hours <= 2 or n_points < 10:
        resample_period = "5min"
    elif time_span_hours <= 24:
        resample_period = "15min"
    elif n_points > 1000:
        resample_period = "1H"
    elif n_points > 200:
        resample_period = "30min"
    else:
        resample_period = "1H"  # default for sparse data

    try:
        ohlc = data["price"].resample(resample_period).agg(["first", "max", "min", "last"]).dropna(how="all")
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc = ohlc.dropna(how="any").reset_index()
        if ohlc.empty or len(ohlc) < 2:
            st.line_chart(data["price"], height=320)
            return

        # Determine color based on price movement (Green = Up, Red = Down)
        ohlc["color"] = ohlc.apply(lambda x: "#22c55e" if x["close"] >= x["open"] else "#ef4444", axis=1)

        if alt is not None and not ohlc.empty:
            # Create interactive selection for zooming
            brush = alt.selection_interval(bind='scales', encodings=['x'])
            
            # Format time axis
            time_axis = alt.X(
                "time:T", 
                title="Time",
                axis=alt.Axis(
                    format="%b %d, %Y %H:%M",
                    labelAngle=-45,
                    tickCount=8
                )
            ).scale(domain=brush)
            
            # Create candlestick chart with zoom
            rules = alt.Chart(ohlc).mark_rule().encode(
                x=time_axis,
                y=alt.Y("low:Q", title="Price ($)"),
                y2="high:Q",
                color=alt.Color("color:N", scale=None),
            ).add_params(brush)

            bars = alt.Chart(ohlc).mark_bar(width=8).encode(
                x=time_axis,
                y="open:Q",
                y2="close:Q",
                color=alt.Color("color:N", scale=None),
                tooltip=[
                    alt.Tooltip("time:T", title="Time", format="%b %d, %Y %H:%M"),
                    alt.Tooltip("open:Q", title="Open", format="$.2f"),
                    alt.Tooltip("high:Q", title="High", format="$.2f"),
                    alt.Tooltip("low:Q", title="Low", format="$.2f"),
                    alt.Tooltip("close:Q", title="Close", format="$.2f"),
                ],
            ).add_params(brush)

            chart = (rules + bars).properties(
                title=f"{symbol} Candlestick Chart ({resample_period} intervals)",
                height=320
            ).configure_axis(
                grid=True,
                gridColor="#374151",
                labelColor="#e5e7eb",
                titleColor="#e5e7eb",
            ).configure_view(
                stroke=None
            ).configure_title(
                color="#e5e7eb"
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(ohlc.set_index("time")["close"], height=320)
    except Exception as e:
        st.warning(f"Could not build candlestick chart: {e}. Showing line chart.")
        try:
            st.line_chart(data["price"], height=320)
        except Exception:
            st.line_chart(data.reset_index().set_index("time")["price"], height=320)


def _render_volume_price_chart(data, symbol):
    """Render combined volume and price chart with zoom"""
    data = data.copy()
    data["time"] = pd.to_datetime(data["time"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price"]).sort_values("time").reset_index(drop=True)
    if "volume" not in data.columns:
        data["volume"] = 0
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0)
    if len(data) < 2:
        st.line_chart(data.set_index("time")["price"], height=260)
        return
    if alt is not None:
        data["price_change"] = data["price"].diff().fillna(0)
        data["color"] = data["price_change"].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444")
        
        src = data.reset_index(drop=True)
        
        # Create interactive selection for zooming
        brush = alt.selection_interval(bind='scales', encodings=['x'])
        
        # Format time axis properly
        time_axis = alt.X(
            "time:T", 
            title="Time",
            axis=alt.Axis(
                format="%b %d, %Y %H:%M",
                labelAngle=-45,
                tickCount=8
            )
        ).scale(domain=brush)

        base = alt.Chart(src)
        price_line = base.mark_line(strokeWidth=3, interpolate='linear').encode(
            x=time_axis,
            y=alt.Y("price:Q", title="Price ($)"),
            color=alt.Color(
                "color:N",
                scale=None,
                legend=None
            ),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %d, %Y %H:%M"),
                alt.Tooltip("price:Q", title="Price", format="$.2f"),
            ],
        ).add_params(brush).properties(height=260)

        # Volume chart (bottom) - shares x-axis with price
        volume_bars = alt.Chart(src).mark_bar(opacity=0.6, color="#3b82f6").encode(
            x=time_axis,
            y=alt.Y("volume:Q", title="Volume", axis=alt.Axis(titleColor="#3b82f6")),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %d, %Y %H:%M"),
                alt.Tooltip("volume:Q", title="Volume", format=".2s"),
            ],
        ).add_params(brush).properties(height=120)

        chart = alt.vconcat(price_line, volume_bars).resolve_scale(
            y="independent"
        ).properties(
            title=f"{symbol} Price & Volume"
        ).configure_axis(
            grid=True,
            gridColor="#374151",
            labelColor="#e5e7eb",
            titleColor="#e5e7eb",
        ).configure_view(
            stroke=None
        ).configure_title(
            color="#e5e7eb"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(data.set_index("time")["price"], height=260)
        if "volume" in data.columns and data["volume"].notna().any():
            st.bar_chart(data.set_index("time")["volume"], height=120)

def _render_cmc_trends_and_corr():
    key = _get_cmc_key()
    top = []
    try:
        top = fetch_cmc_listings(limit=5)
    except Exception:
        top = []
    if not top:
        top = _placeholder_live_data()
    symbols = [d["symbol"] for d in top][:5]
    ids = {d["symbol"]: d.get("id") for d in top if d.get("id") is not None}
    end = datetime.now()
    start = end - pd.Timedelta(days=30)
    frames = []
    for sym in symbols:
        df = pd.DataFrame()
        cmc_id = ids.get(sym)
        if key and cmc_id:
            try:
                df = fetch_cmc_ohlcv_history(cmc_id, start.isoformat(), end.isoformat())
                if not df.empty:
                    df = df.rename(columns={"close": "price"})
                    df["symbol"] = sym
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            try:
                gecko_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "ADA": "cardano"}
                gid = gecko_map.get(sym)
                if gid:
                    gdf = fetch_price_history(gid, days=30)
                    if not gdf.empty:
                        gdf = gdf.reset_index().rename(columns={"time": "time"})
                        gdf["symbol"] = sym
                        df = gdf[["time", "price", "symbol"]]
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            syn = _synthetic_history([sym], 240)
            syn = syn.rename(columns={"time": "time"})
            df = syn[["time", "price", "symbol"]]
        frames.append(df)
    if not frames:
        st.info("No data available for CMC trends.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    if alt is not None and not all_df.empty:
        sel = st.multiselect("Select assets", sorted(symbols), default=symbols)
        plot_df = all_df[all_df["symbol"].isin(sel)].copy()
        plot_df["time"] = pd.to_datetime(plot_df["time"])
        line = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("price:Q", title="Price ($)"),
            color="symbol:N",
            tooltip=[alt.Tooltip("time:T", format="%b %d, %Y"), "symbol:N", alt.Tooltip("price:Q", format="$.2f")],
        ).properties(height=260)
        st.altair_chart(line, use_container_width=True)
    else:
        pivot = all_df.pivot_table(index="time", columns="symbol", values="price")
        st.line_chart(pivot)
    pivot = all_df.copy()
    pivot = pivot.pivot_table(index="time", columns="symbol", values="price")
    returns = pivot.pct_change().dropna(how="any")
    if returns.empty:
        st.info("Not enough data for correlation.")
        return
    corr = returns.corr()
    st.write("Correlation matrix (daily returns)")
    if alt is not None:
        corr_reset = corr.reset_index()
        id_col = corr_reset.columns[0]
        corr_long = corr_reset.rename(columns={id_col: "symbol"}).melt(id_vars="symbol", var_name="symbol2", value_name="corr")
        heat = alt.Chart(corr_long).mark_rect().encode(
            x=alt.X("symbol:N", title=""),
            y=alt.Y("symbol2:N", title=""),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
            tooltip=["symbol:N", "symbol2:N", alt.Tooltip("corr:Q", format=".2f")],
        ).properties(height=220)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), use_container_width=True)

st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("page", "welcome")
if st.session_state["authenticated"]:
    if st.session_state.get("page") == "intro":
        _show_intro()
    elif st.session_state.get("page") == "milestone2":
        _show_milestone2_dashboard()
    elif st.session_state.get("page") == "milestone3":
        _show_milestone3_dashboard()
    elif st.session_state.get("page") == "milestone4":
        _show_milestone4_dashboard()
    else:
        st.session_state["page"] = "dashboard"
        _show_dashboard()
else:
    if st.session_state["page"] == "welcome":
        _show_welcome()
    elif st.session_state["page"] == "login":
        _show_login()
    elif st.session_state["page"] == "signup":
        _show_signup()
    else:
        _show_welcome()
