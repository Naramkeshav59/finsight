import time
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup


def load_config(config_path: Path) -> dict:
    # reads configs/config.yaml and returns it as a python dict
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_filings(cik: str, years: list[int], headers: dict) -> list[dict]:
    # hits the EDGAR submissions endpoint for a company and returns
    # only the 10-K filings that match the requested years
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    # EDGAR's "recent" section only holds the latest ~40 filings. High-volume filers
    # (like JPM with dozens of 8-Ks per year) push older 10-Ks into paginated "files".
    # We collect all batches so we never miss a filing regardless of filing frequency.
    batches = [data["filings"]["recent"]]
    for file_ref in data["filings"].get("files", []):
        batch_url = f"https://data.sec.gov/submissions/{file_ref['name']}"
        batch_resp = requests.get(batch_url, headers=headers)
        batch_resp.raise_for_status()
        batches.append(batch_resp.json())
        time.sleep(0.1)

    filings = []
    for batch in batches:
        for form, accession, report_date, doc in zip(
            batch["form"],
            batch["accessionNumber"],
            batch["reportDate"],   # fiscal year end date, e.g. "2023-12-31"
            batch["primaryDocument"],
        ):
            # use reportDate (not filingDate) so companies with Dec 31 fiscal year
            # ends (whose 10-Ks are filed in Feb of the *next* year) still match
            if form == "10-K" and int(report_date[:4]) in years:
                filings.append({
                    "accession": accession,
                    "date": report_date,
                    "year": int(report_date[:4]),
                    "primary_doc": doc,
                })

    return filings


def fetch_document(cik: str, filing: dict, headers: dict) -> str:
    # builds the EDGAR archive URL, fetches the 10-K HTML, strips tags,
    # and returns clean plain text
    accession_clean = filing["accession"].replace("-", "")  # dashes removed for URL
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_clean}/{filing['primary_doc']}"
    )
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def ingest(config_path: Path = Path("configs/config.yaml")) -> None:
    # main orchestrator: loops over every company and year in the config,
    # downloads the 10-K if not already on disk, saves as plain text
    config = load_config(config_path)

    headers = {"User-Agent": "FinSight naramkeshav59@gmail.com"}

    for company in config["companies"]:
        ticker = company["ticker"]
        cik = company["cik"]
        years = company["years"]

        print(f"\n{ticker} — fetching filing list...")
        filings = fetch_filings(cik, years, headers)
        print(f"  found {len(filings)} 10-K filing(s)")

        out_dir = Path(config["paths"]["raw_data"]) / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        for filing in filings:
            out_path = out_dir / f"{filing['year']}.txt"
            if out_path.exists():
                print(f"  {filing['year']} already exists, skipping")
                continue

            print(f"  fetching {filing['year']} ({filing['date']})...")
            text = fetch_document(cik, filing, headers)

            out_path.write_text(text, encoding="utf-8")
            print(f"  saved {out_path} ({len(text):,} chars)")

            time.sleep(0.5)  # be polite to SEC servers


if __name__ == "__main__":
    ingest()
