import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional

from PyQt5 import QtCore

GITHUB_REPO = "JakubHekal/IntanRHXLiveFeed"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"


@dataclass
class UpdateInfo:
    available: bool
    latest_version: str = ""
    current_version: str = ""
    download_url: Optional[str] = None
    release_url: str = RELEASES_URL


def _version_tuple(v: str) -> tuple:
    match = re.search(r"(\d+(?:\.\d+)*)", v)
    if not match:
        return (0,)
    return tuple(int(x) for x in match.group(1).split("."))


def _do_check(current_version: str) -> UpdateInfo:
    req = urllib.request.Request(
        API_URL,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": f"RHX-Realtime-Feed/{current_version}",
        },
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())

    latest_tag = data.get("tag_name", "")
    latest = latest_tag.lstrip("vV")

    download_url = None
    for asset in data.get("assets", []):
        if asset.get("name", "").endswith(".exe"):
            download_url = asset.get("browser_download_url")
            break

    available = _version_tuple(latest) > _version_tuple(current_version)
    return UpdateInfo(
        available=available,
        latest_version=latest,
        current_version=current_version,
        download_url=download_url,
        release_url=data.get("html_url", RELEASES_URL),
    )


class UpdateCheckThread(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)

    def __init__(self, current_version: str, parent=None):
        super().__init__(parent)
        self._current = current_version

    def run(self):
        try:
            info = _do_check(self._current)
            self.result_ready.emit(info)
        except Exception as e:
            self.result_ready.emit(e)
