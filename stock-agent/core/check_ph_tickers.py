"""Standalone check: print every ticker phisix's ALL-stocks endpoint currently returns.

Doesn't import anything from the app itself - just hits phisix directly so you can see
exactly what the raw data source has, independent of any caching/fallback logic in ph_data.py.

Run: python list_phisix_tickers.py
"""
import requests

HOSTS = [
    "https://phisix-api3.appspot.com",
    "https://phisix-api.appspot.com",
    "https://phisix-api2.appspot.com",
]


def main():
    for host in HOSTS:
        print(f"Trying {host} ...")
        try:
            resp = requests.get(f"{host}/stocks.json", timeout=20)
        except requests.RequestException as e:
            print(f"  failed: {e}\n")
            continue

        print(f"  HTTP {resp.status_code}")
        if resp.status_code != 200:
            print()
            continue

        try:
            data = resp.json()
            stocks = data["stocks"]
        except (ValueError, KeyError, TypeError) as e:
            print(f"  unexpected response shape: {e}\n")
            continue

        print(f"  as_of: {data.get('as_of')}")
        print(f"  total tickers: {len(stocks)}\n")

        for s in sorted(stocks, key=lambda x: x.get("symbol", "")):
            symbol = s.get("symbol", "?")
            name = s.get("name", "?")
            print(f"{symbol:<8} {name}")

        return  # stop after the first host that actually works

    print("All hosts failed - phisix may be down, or all mirrors are unreachable right now.")


if __name__ == "__main__":
    main()