import os
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
import requests
import pandas as pd
from datetime import datetime

app = Flask(__name__)
# Use an environment variable in production.
# (A default is kept so the demo runs locally without setup.)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

# ---------------- LOGIN CREDENTIALS ----------------
USERNAME = "admin@gmail.com"
PASSWORD = "123456"

# ---------------- CRYPTO FETCH ----------------
def fetch_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 5,
        "page": 1,
        "sparkline": False
    }

    response = requests.get(url, params=params)
    data = response.json()

    crypto_list = []
    for coin in data:
        crypto_list.append({
            "name": coin["name"],
            "symbol": coin["symbol"].upper(),
            "price": coin["current_price"],
            "change": coin["price_change_percentage_24h"],
            "volume": coin["total_volume"]
        })

    df = pd.DataFrame(crypto_list)
    df.fillna(0, inplace=True)
    df["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv("crypto_data.csv", index=False)

    return crypto_list

# ---------------- ROUTES ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/data")
def data():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    crypto_data = fetch_crypto_data()
    return jsonify(crypto_data)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    # Note:
    # - use_reloader=True uses the 'signal' module, which only works
    #   in the main thread of the main interpreter.
    # - This breaks in environments like Streamlit or some IDE runners.
    #
    # To avoid "ValueError: signal only works in main thread of the main interpreter",
    # we disable the reloader here.
    app.run(debug=True, use_reloader=False)