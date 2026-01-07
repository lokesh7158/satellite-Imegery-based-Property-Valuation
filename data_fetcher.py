import os
import time
import requests
import pandas as pd
from pathlib import Path

CLIENT_ID = "53e190be-92a4-413f-9e2b-e88fc2adbe3c"
CLIENT_SECRET = "CXmCWK8BJEVEcJajnYqsLv05MBr3aIho"

INPUT_TRAIN = r"C:\Users\lokes\train_raw.xlsx"
INPUT_TEST  = r"C:\Users\lokes\test.xlsx"

IMAGE_DIR = r"C:\Users\lokes\Downloads\data\images"
META_OUT_TRAIN = r"C:\Users\lokes\metadata_train.csv"
META_OUT_TEST  = r"C:\Users\lokes\metadata_test.csv"


DELTA = 0.005          # ~500m around property
IMG_SIZE = 768         # higher resolution

TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

TOKEN_EXPIRES_IN = 3500  # seconds (~58 min)

os.makedirs(IMAGE_DIR, exist_ok=True)

_token = None
_token_time = 0


def get_access_token():
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def get_valid_token():
    global _token, _token_time

    if _token is None or (time.time() - _token_time) > TOKEN_EXPIRES_IN:
        _token = get_access_token()
        _token_time = time.time()
        print("🔑 Access token refreshed")

    return _token


def create_bbox(lat, lon, delta):
    return [lon - delta, lat - delta, lon + delta, lat + delta]


EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(s) {
  if (s.dataMask === 0) {
    return [0, 0, 0, 0];
  }

  return [
    Math.min(1.0, s.B04 * 3.0),
    Math.min(1.0, s.B03 * 3.0),
    Math.min(1.0, s.B02 * 3.0),
    1
  ];
}
"""


def fetch_image(lat, lon, out_path, retries=1):
    token = get_valid_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "bounds": {
                "bbox": create_bbox(lat, lon, DELTA),
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": "2023-01-01T00:00:00Z",
                            "to": "2023-12-31T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": IMG_SIZE,
            "height": IMG_SIZE,
            "responses": [
                {"identifier": "default", "format": {"type": "image/png"}}
            ],
        },
        "evalscript": EVALSCRIPT,
    }

    r = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=60)

    if r.status_code == 401 and retries > 0:
        print("🔄 Token expired, retrying...")
        global _token
        _token = None
        return fetch_image(lat, lon, out_path, retries - 1)

    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)


def process_dataset(df, label):
    df = df.dropna(subset=["lat", "long"]).reset_index(drop=True)

    img_paths = []

    for i, row in df.iterrows():
        house_id = str(row["id"]).zfill(10)
        img_path = Path(IMAGE_DIR) / f"{house_id}.png"

        if img_path.exists():
            img_paths.append(str(img_path))
            print(f"[{i+1}/{len(df)}] {label}: skipped")
            continue

        try:
            fetch_image(
                lat=row["lat"],
                lon=row["long"],
                out_path=img_path,
            )
            img_paths.append(str(img_path))
            print(f"[{i+1}/{len(df)}] {label}: saved {img_path}")

            time.sleep(0.2)  

        except Exception as e:
            img_paths.append("")
            print(f"[{i+1}/{len(df)}] {label}: failed id={house_id}: {e}")

    df["img_path"] = img_paths
    return df

def main():
    train_df = pd.read_excel(INPUT_TRAIN)
    test_df = pd.read_excel(INPUT_TEST)

    train_df = process_dataset(train_df, "TRAIN")
    test_df = process_dataset(test_df, "TEST")

    train_df.to_csv(META_OUT_TRAIN, index=False)
    test_df.to_csv(META_OUT_TEST, index=False)

    print("\n📁 Metadata saved:")
    print(f" → {META_OUT_TRAIN}")
    print(f" → {META_OUT_TEST}")


if __name__ == "__main__":
    main()
