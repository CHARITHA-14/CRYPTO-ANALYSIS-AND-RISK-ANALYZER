import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crypto Analysis & Risk Analyzer",
    page_icon="💹",
    layout="wide",
)


# ---------------- LOGIN CREDENTIALS ----------------
USERNAME = "admin@gmail.com"
PASSWORD = "123456"


# ---------------- STYLING HELPERS ----------------
def inject_login_css() -> None:
    """
    Inject the existing login.css styles into Streamlit so the
    login view keeps your custom look & animations.

    IMPORTANT:
    Streamlit manages the <body> layout. Your existing CSS styles `body` as a flex
    container, which can cause a blank/empty screen in Streamlit. We safely scope
    those rules to Streamlit's root container (`.stApp`) instead.
    """
    css_path = BASE_DIR / "static" / "login.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")

        # Scope potentially-breaking selectors away from <body>
        css = css.replace("body::before", ".stApp::before")
        css = css.replace("body > *", ".stApp > *")
        css = css.replace("body{", ".stApp{")
        css = css.replace("body {", ".stApp {")

        # Make sure Streamlit default chrome doesn't overlay the background
        css += """
        /* Streamlit layout fixes */
        .stApp { min-height: 100vh; }
        section.main > div { padding-top: 2rem; }
        """

        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------- DATA FETCH ----------------
def fetch_crypto_data() -> pd.DataFrame:
    """
    Fetch crypto data from CoinGecko and return as a DataFrame.
    Also writes a CSV similar to the original Flask app.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 5,
        "page": 1,
        "sparkline": False,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    crypto_list = []
    for coin in data:
        crypto_list.append(
            {
                "Name": f"{coin['name']} ({coin['symbol'].upper()})",
                "Price ($)": coin["current_price"],
                "24h %": coin["price_change_percentage_24h"],
                "Volume": coin["total_volume"],
            }
        )

    df = pd.DataFrame(crypto_list)
    df.fillna(0, inplace=True)
    df["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv("crypto_data.csv", index=False)
    return df


# ---------------- VIEWS ----------------
def show_login():
    inject_login_css()

    # Ensure state keys exist
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("login_error", "")

    # Page heading (helps confirm the app is rendering even if CSS breaks)
    st.markdown("## Crypto Analysis & Risk Analyzer")

    # Build a layout that reuses your classes
    st.markdown('<div class="login-page">', unsafe_allow_html=True)

    # Illustration column
    col_ill, col_form = st.columns([1.1, 1])

    with col_ill:
        st.markdown('<div class="login-illustration">', unsafe_allow_html=True)
        img_path_svg = BASE_DIR / "static" / "login-graphic.svg"
        if img_path_svg.exists():
            st.image(str(img_path_svg), use_container_width=True)
        else:
            st.info("Add an image at `static/login-graphic.svg` to see the illustration.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_form:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("## Crypto Login")

        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if username == USERNAME and password == PASSWORD:
                st.session_state["authenticated"] = True
                st.session_state["login_error"] = ""
                st.rerun()
            else:
                st.session_state["authenticated"] = False
                st.session_state["login_error"] = "Invalid credentials"

        if st.session_state.get("login_error"):
            st.markdown(
                f'<div class="error">{st.session_state["login_error"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def show_dashboard():
    st.sidebar.title("Crypto Analyzer")
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["login_error"] = ""
        st.rerun()

    st.title("Milestone 1 – Data Acquisition")
    st.caption("Live crypto prices via CoinGecko with CSV logging.")

    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("Requirements")
        st.markdown(
            """
            - Python Environment Setup  
            - API Integration (CoinGecko)  
            - CSV Data Storage  
            - Missing Value Handling  
            """
        )

        st.subheader("Outputs")
        st.markdown(
            """
            - Live Crypto Prices  
            - Verified API Connectivity  
            - Auto CSV Storage  
            """
        )

    with col_right:
        top_row = st.columns([1, 0.4])
        with top_row[0]:
            st.subheader("Crypto Data Fetcher · Live")
        with top_row[1]:
            refresh = st.button("Refresh data")

        # Fetch data either on first load or when refresh is clicked
        if "crypto_df" not in st.session_state or refresh:
            try:
                st.session_state["crypto_df"] = fetch_crypto_data()
            except Exception as e:
                st.error(f"Error fetching data from CoinGecko: {e}")
                return

        df = st.session_state["crypto_df"].copy()

        if not df.empty:
            last_time = df["time"].iloc[0]
            st.caption(f"Last updated: {last_time}")

            # Display a table similar to the HTML version
            display_df = df[["Name", "Price ($)", "24h %", "Volume"]]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            # Simple risk cues based on 24h change
            st.markdown("### Quick Risk Glimpse")
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            avg_change = display_df["24h %"].mean()
            top_gain = display_df.sort_values("24h %", ascending=False).iloc[0]
            top_loss = display_df.sort_values("24h %", ascending=True).iloc[0]

            with risk_col1:
                st.metric("Average 24h change", f"{avg_change:.2f}%")
            with risk_col2:
                st.metric("Top gainer", top_gain["Name"], f"{top_gain['24h %']:.2f}%")
            with risk_col3:
                st.metric("Top loser", top_loss["Name"], f"{top_loss['24h %']:.2f}%")


# ---------------- MAIN ENTRYPOINT ----------------
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        show_login()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()

