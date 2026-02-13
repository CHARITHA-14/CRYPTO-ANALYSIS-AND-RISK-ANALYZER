"""
Crypto Analysis and Risk Analyzer — single project.
Run: streamlit run app.py
"""
import json
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
import random
import os
import hashlib
try:
    import altair as alt
except Exception:
    alt = None

def _init_altair():
    if alt is None:
        return
    try:
        alt.data_transformers.disable_max_rows()
    except Exception:
        pass

# ---------------- CONFIG & DATA ----------------
USERNAME = "admin@gmail.com"
PASSWORD = "123456"
BASE_DIR = Path(__file__).resolve().parent
USER_DATA_FILE = BASE_DIR / "user_added_data.json"
HISTORY_FILE = BASE_DIR / "history.csv"
ACCOUNTS_FILE = BASE_DIR / "user_accounts.json"


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

def fetch_cmc_listings(limit=5):
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
    times = pd.date_range(end=end, periods=points, freq="H")
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


def _inject_css():
    css_path = BASE_DIR / "static" / "login.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        css = css.replace("body::before", ".stApp::before").replace("body > *", ".stApp > *")
        css = css.replace("body{", ".stApp{").replace("body {", ".stApp {")
        css += "\n.stApp { min-height: 100vh; }\n"
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def _inject_welcome_css():
    css_path = BASE_DIR / "static" / "welcome.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        css = css.replace("body::before", ".stApp::before").replace("body > *", ".stApp > *")
        css = css.replace("body{", ".stApp{").replace("body {", ".stApp {")
        css += "\n.stApp { min-height: 100vh; }\n"
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def _show_welcome():
    _inject_welcome_css()
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
    _inject_css()
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
            "Crypto Analysis &amp; Risk Analyzer</p>"
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
                st.session_state["page"] = "dashboard"
                st.rerun()
            else:
                st.session_state["login_error"] = "Invalid credentials"
        if st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])

def _show_signup():
    _inject_css()
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
                    st.session_state["page"] = "dashboard"
                    st.rerun()


def _show_dashboard():
    # optional dashboard-specific styling
    css_path = BASE_DIR / "static" / "dashboard.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        css = css.replace("body::before", ".stApp::before").replace("body > *", ".stApp > *")
        css = css.replace("body{", ".stApp{").replace("body {", ".stApp {")
        css += "\n.stApp { min-height: 100vh; }\n"
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.sidebar.title("Crypto Volatility and Risk Analyzer")
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["login_error"] = ""
        st.rerun()

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
          <h1 style="margin:0;font-size:2rem;font-weight:700;letter-spacing:-0.02em;color:#e5e7eb;">
            Cryptocurrency & Risk Concepts
          </h1>
          <p style="margin:6px 0 0 0;font-size:0.95rem;color:#94a3b8;">
            Crypto markets are highly volatile and driven by liquidity, sentiment, and macro risk. 
            Monitor price swings, volume, and drawdowns to assess exposure. Diversify across assets, 
            track correlations, and set position limits to manage downside risk.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # (Charts moved below the table per request.)

    col_left, col_right = st.columns([1, 1.6])
    with col_left:
        st.subheader("📘 What is Cryptocurrency?")
        st.markdown(
            "Cryptocurrency is digital money that exists online and runs on blockchain technology, "
            "without control from banks or governments. Coins like Bitcoin and Ethereum can be used "
            "for trading, investing, and online transactions. Their prices change frequently based on "
            "demand, news, and market activity."
        )

        st.subheader("📊 What This Dashboard Shows")
        st.markdown(
            "This dashboard provides a simple view of the crypto market with live price updates, "
            "top gainers and losers, and key movement indicators. It helps users quickly understand "
            "which coins are rising, falling, and showing high volatility."
        )

        st.subheader("🚀 Why It's Useful")
        st.markdown(
            "It helps beginners and traders track market trends, spot sudden price changes, and "
            "understand risk through real-time data. Everything is presented in one place to make "
            "crypto market analysis easy and quick."
        )

    with col_right:
        # Header row with title and refresh button
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.subheader("Crypto Data Fetcher · Live")
        with header_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh", key="refresh_data_btn", use_container_width=True):
                st.session_state["refresh_count"] += 1
                # Force data refresh by clearing any cached values
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

    # ---------------- HISTORIC DATA VISUALIZATION SECTION ----------------
    st.markdown("---")
    st.subheader("Historic Data Visualization")
    st.caption("Visualize price trends from stored history data over time.")

    _render_historic_visualization()


def _render_historic_visualization():
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

    # Controls for visualization
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_symbol = st.selectbox("Select cryptocurrency", symbols, key="hist_viz_symbol")
    with col2:
        time_range = st.selectbox(
            "Time range",
            ["All time", "Last 24 hours", "Last 7 days", "Last 30 days"],
            key="hist_viz_range"
        )
    with col3:
        chart_type = st.selectbox(
            "Chart type",
            ["Price Line", "Candlestick-style", "Volume + Price"],
            key="hist_viz_chart_type"
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

    # Prepare data
    symbol_data["price"] = pd.to_numeric(symbol_data["price"], errors="coerce")
    symbol_data["volume"] = pd.to_numeric(symbol_data["volume"], errors="coerce")
    symbol_data["change"] = pd.to_numeric(symbol_data["change"], errors="coerce")

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
        try:
            _render_price_line_chart(symbol_data, selected_symbol)
        except Exception:
            st.line_chart(symbol_data.set_index("time")["price"], height=400)
    elif chart_type == "Candlestick-style":
        try:
            _render_candlestick_chart(symbol_data, selected_symbol)
        except Exception:
            st.line_chart(symbol_data.set_index("time")["price"], height=400)
    else:
        try:
            _render_volume_price_chart(symbol_data, selected_symbol)
        except Exception:
            st.line_chart(symbol_data.set_index("time")["price"], height=400)

    # Show data table
    with st.expander("View Raw Data"):
        display_df = symbol_data[["time", "name", "symbol", "price", "change", "volume"]].copy()
        display_df.columns = ["Time", "Name", "Symbol", "Price ($)", "24h Change (%)", "Volume"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button for filtered data
        csv_data = symbol_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {selected_symbol} History CSV",
            data=csv_data,
            file_name=f"{selected_symbol.lower()}_history.csv",
            mime="text/csv",
            key="dl_hist_viz"
        )


def _render_price_line_chart(data, symbol):
    """Render a zoomable price line chart with directional color scheme"""
    if alt is not None:
        data = data.copy()
        data["time"] = pd.to_datetime(data["time"])
        data["price"] = pd.to_numeric(data["price"], errors="coerce")
        data["price_change"] = data["price"].diff()
        data["rolling_mean"] = data["price"].rolling(window=5, min_periods=1).mean()
        data["rolling_std"] = data["price"].rolling(window=5, min_periods=1).std()
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

        chart = (band + price_line + mean_line + peak_points + valley_points).properties(
            title=f"{symbol} Price History (Green=Up, Red=Down)",
            height=450
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
        st.line_chart(data.set_index("time")["price"], height=450)


def _render_candlestick_chart(data, symbol):
    """Render a candlestick-style chart using OHLC aggregation"""
    data = data.copy()
    data["time"] = pd.to_datetime(data["time"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    # Resample data to create OHLC (Open, High, Low, Close) for time periods
    data = data.set_index("time")

    # Determine resampling period based on data size
    n_points = len(data)
    if n_points > 1000:
        resample_period = "1H"  # 1 hour
    elif n_points > 200:
        resample_period = "30min"  # 30 minutes
    else:
        resample_period = "10min"  # 10 minutes

    try:
        ohlc = data["price"].resample(resample_period).agg(["first", "max", "min", "last"]).dropna()
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc = ohlc.reset_index()

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
                height=400
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
            st.line_chart(ohlc.set_index("time")["close"], height=400)
    except Exception as e:
        st.error(f"Error creating candlestick chart: {e}")
        st.line_chart(data["price"], height=400)


def _render_volume_price_chart(data, symbol):
    """Render combined volume and price chart with zoom"""
    if alt is not None:
        data = data.copy()
        data["time"] = pd.to_datetime(data["time"])
        data["price"] = pd.to_numeric(data["price"], errors="coerce")
        data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
        data["price_change"] = data["price"].diff()
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
        ).add_params(brush).properties(height=300)

        # Volume chart (bottom) - shares x-axis with price
        volume_bars = alt.Chart(src).mark_bar(opacity=0.6, color="#3b82f6").encode(
            x=time_axis,
            y=alt.Y("volume:Q", title="Volume", axis=alt.Axis(titleColor="#3b82f6")),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %d, %Y %H:%M"),
                alt.Tooltip("volume:Q", title="Volume", format=".2s"),
            ],
        ).add_params(brush).properties(height=150)

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
        st.line_chart(data.set_index("time")["price"], height=300)
        if "volume" in data.columns and data["volume"].notna().any():
            st.bar_chart(data.set_index("time")["volume"], height=150)

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
        ).properties(height=300)
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
        corr_long = corr.reset_index().melt(id_vars="symbol", var_name="symbol2", value_name="corr").rename(columns={"index": "symbol"})
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
