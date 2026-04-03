import os
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from .core import ROOT_DIR, err, info, ok, warn

_OS = platform.system().lower()                     # "windows" | "linux" | "darwin"
_ADB_EXE = "adb.exe" if _OS == "windows" else "adb"

LOCAL_ADB_DIR = ROOT_DIR / "platform-tools"
LOCAL_ADB_EXE = LOCAL_ADB_DIR / _ADB_EXE

_DOWNLOAD_URLS = {
    "windows": "https://dl.google.com/android/repository/platform-tools-latest-windows.zip",
    "linux":   "https://dl.google.com/android/repository/platform-tools-latest-linux.zip",
    "darwin":  "https://dl.google.com/android/repository/platform-tools-latest-darwin.zip",
}
DOWNLOAD_URL = _DOWNLOAD_URLS.get(_OS, _DOWNLOAD_URLS["linux"])
MIRROR_URL = ""


def adb_works(adb_path: str) -> bool:
    try:
        result = subprocess.run(
            [adb_path, "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and "Android Debug Bridge" in result.stdout
    except Exception:
        return False


def find_adb() -> Optional[str]:
    if LOCAL_ADB_EXE.exists() and adb_works(str(LOCAL_ADB_EXE)):
        return str(LOCAL_ADB_EXE)
    adb_in_path = shutil.which("adb")
    if adb_in_path and adb_works(adb_in_path):
        return adb_in_path
    return None


def download_adb() -> None:
    import requests

    if LOCAL_ADB_DIR.exists():
        warn("检测到旧的 platform-tools 目录，清理后重新下载...")
        shutil.rmtree(LOCAL_ADB_DIR)

    zip_path = ROOT_DIR / "platform-tools.zip"
    sources = [(MIRROR_URL, "自定义镜像")] if MIRROR_URL else []
    sources.append((DOWNLOAD_URL, "Google CDN"))

    for url, label in sources:
        info(f"尝试从 {label} 下载...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            with open(zip_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        handle.write(chunk)
                        received += len(chunk)
                        if total:
                            pct = received / total * 100
                            print(
                                f"\r    platform-tools.zip  {received/1024/1024:.1f}/{total/1024/1024:.1f} MB  {pct:.0f}%",
                                end="",
                                flush=True,
                            )
            print()
            ok(f"{label} 下载完成")
            break
        except Exception as exc:
            print()
            warn(f"{label} 失败：{exc}")
    else:
        err("所有下载源均失败，请手动下载：")
        err(f"  {DOWNLOAD_URL}")
        err(f"  解压后将 platform-tools/ 放到：{ROOT_DIR}")
        raise SystemExit(1)

    info("解压 platform-tools.zip ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(ROOT_DIR)
    zip_path.unlink(missing_ok=True)

    if not LOCAL_ADB_EXE.exists():
        err(f"解压后未找到 {LOCAL_ADB_EXE}")
        raise SystemExit(1)

    if _OS != "windows":
        os.chmod(LOCAL_ADB_EXE, 0o755)
        info(f"已设置权限 chmod +x: {LOCAL_ADB_EXE}")

    ok(f"ADB 解压完成：{LOCAL_ADB_EXE}")


def ensure_adb() -> str:
    print("\n[ADB 检测]")
    adb = find_adb()
    if adb:
        ok(f"ADB 已就绪：{adb}")
        return adb
    warn("未找到可用的 ADB，开始自动下载...")
    download_adb()
    adb = str(LOCAL_ADB_EXE)
    if not adb_works(adb):
        err("ADB 安装后仍无法运行，请检查")
        raise SystemExit(1)
    ok(f"ADB 安装完成：{adb}")
    return adb
