"""Download the German Credit dataset from the UCI repository into data/raw/.

Run from the project root:
    python -m src.download_data
"""

import io
import logging
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
)
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REQUIRED_FILES = ["german.data", "german.doc"]


def download_dataset() -> bool:
    """Download the zip, extract the required files, and return True on success."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if all((RAW_DIR / name).exists() for name in REQUIRED_FILES):
        logger.info("Dataset already present in %s. Nothing to do.", RAW_DIR)
        return True

    logger.info("Downloading dataset from UCI...")
    try:
        request = Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            zip_bytes = response.read()
    except (URLError, TimeoutError) as error:
        logger.error("Download failed: %s", error)
        logger.error(
            "Please download manually: open "
            "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data, "
            "click Download, and extract german.data and german.doc into %s",
            RAW_DIR,
        )
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            available = archive.namelist()
            for name in REQUIRED_FILES:
                if name not in available:
                    logger.error("%s not found inside the zip. Found: %s", name, available)
                    return False
                (RAW_DIR / name).write_bytes(archive.read(name))
                logger.info("Saved %s", RAW_DIR / name)
    except zipfile.BadZipFile:
        logger.error("Downloaded file is not a valid zip. Try the manual download.")
        return False

    return True


if __name__ == "__main__":
    sys.exit(0 if download_dataset() else 1)