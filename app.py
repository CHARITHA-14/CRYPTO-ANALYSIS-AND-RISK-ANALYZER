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

# ---------------- CONFIG & DATA ----------------
USERNAME = "admin@gmail.com"
PASSWORD = "123456"
BASE_DIR = Path(__file__).resolve().parent
USER_DATA_FILE = BASE_DIR / "user_added_data.json"


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


def fetch_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 5,
        "page": 1,
        "sparkline": False,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    crypto_list = []
    for coin in data:
        crypto_list.append({
            "name": coin["name"],
            "symbol": coin["symbol"].upper(),
            "price": coin["current_price"],
            "change": coin["price_change_percentage_24h"] if coin.get("price_change_percentage_24h") is not None else 0,
            "volume": coin["total_volume"],
            "source": "api",
        })
    return crypto_list


def get_combined_data():
    api_data = []
    try:
        api_data = fetch_crypto_data()
    except (requests.RequestException, ValueError):
        pass
    for u in load_user_data():
        api_data.append({
            "name": u.get("name", ""),
            "symbol": u.get("symbol", "").upper(),
            "price": float(u.get("price", 0)),
            "change": float(u.get("change", 0)),
            "volume": float(u.get("volume", 0)),
            "source": "user",
        })
    return api_data


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


# ---------------- STREAMLIT UI ----------------
st.set_page_config(
    page_title="Crypto Analysis and Risk Analyzer",
    page_icon="💹",
    layout="wide",
)


def _inject_css():
    css_path = BASE_DIR / "static" / "login.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        css = css.replace("body::before", ".stApp::before").replace("body > *", ".stApp > *")
        css = css.replace("body{", ".stApp{").replace("body {", ".stApp {")
        css += "\n.stApp { min-height: 100vh; }\n"
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _show_login():
    _inject_css()
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("login_error", "")
    col_ill, col_form = st.columns([1.1, 1])
    with col_ill:
        img_path = BASE_DIR / "static" / "login-graphic.svg"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
    with col_form:
        st.markdown(
            "<p style='margin:0 0 6px 0;font-size:0.75rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#64748b'>Crypto Analysis and Risk Analyzer</p><h2 style='margin:0 0 20px 0;font-size:1.35rem;font-weight:600;color:#0f172a'>Sign in</h2>",
            unsafe_allow_html=True,
        )
        username = st.text_input("Username", key="su")
        password = st.text_input("Password", type="password", key="sp")
        if st.button("Sign in"):
            if username == USERNAME and password == PASSWORD:
                st.session_state["authenticated"] = True
                st.session_state["login_error"] = ""
                st.rerun()
            else:
                st.session_state["login_error"] = "Invalid credentials"
        if st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])


def _show_dashboard():
    st.sidebar.title("Crypto Analysis and Risk Analyzer")
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["login_error"] = ""
        st.rerun()

    # Classic realtime dashboard header
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
          <h1 style="margin:0;font-size:1.8rem;font-weight:600;letter-spacing:-0.02em;">
            Realtime Crypto Risk Dashboard
          </h1>
          <p style="margin:4px 0 0 0;font-size:0.9rem;color:#64748b;">
            Live market snapshot with key risk signals and your own custom entries.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("---")

    col_left, col_right = st.columns([1, 1.6])
    with col_left:
        st.subheader("Add your data")
        st.caption("Log custom tokens, simulated scenarios, or watchlist entries.")
        with st.form("add_data_form"):
            an = st.text_input("Name", key="an")
            sy = st.text_input("Symbol", key="sy")
            pr = st.number_input("Price ($)", min_value=0.0, value=0.0, step=0.01, key="pr")
            ch = st.number_input("24h %", value=0.0, step=0.01, key="ch")
            vo = st.number_input("Volume", min_value=0.0, value=0.0, step=1.0, key="vo")
            if st.form_submit_button("Add entry"):
                if an and sy:
                    entries = load_user_data()
                    entries.append({"name": an, "symbol": sy.upper(), "price": pr, "change": ch, "volume": vo})
                    save_user_data(entries)
                    st.success("Entry added.")
                    st.rerun()
                else:
                    st.error("Name and symbol required.")

        st.subheader("How realtime users benefit")
        st.markdown(
            "- Watch API prices and manual entries together.\n"
            "- Use 24h % and volume as quick risk indicators.\n"
            "- Refresh happens on every interaction; add entries while the market moves."
        )

    with col_right:
        st.subheader("Live market & custom entries")
        if data_all:
            df = pd.DataFrame(data_all)
            df["volume"] = df["volume"].apply(
                lambda x: f\"{x/1e9:.2f}B\" if x >= 1e9 else f\"{x/1e6:.2f}M\" if x >= 1e6 else str(x)
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
        else:
            st.info("No data yet. Add an entry or try again in a moment.")


st.session_state.setdefault("authenticated", False)
if not st.session_state["authenticated"]:
    _show_login()
else:
    _show_dashboard()
