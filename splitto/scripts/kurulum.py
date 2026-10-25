"""
Stem ayırma ortamını kurar (bir kez çalıştırılır, sonra tekrar gerekmez).

- Kullanıcının ev klasöründe kısa yollu bir Python ortamı açar: ~/.stems-venv
  (Windows'ta uzun yollar CUDA DLL'lerini yükletmiyor, bu yüzden kısa yol şart.)
- audio-separator'ı kurar; NVIDIA ekran kartı varsa CUDA'lı PyTorch'u üstüne koyar.
- ffmpeg var mı bakar.

Kullanım:  python kurulum.py            # kurulu ise sadece kontrol eder
           python kurulum.py --yeniden  # ortamı silip baştan kurar
Ortam yolunu değiştirmek için STEMS_VENV ortam değişkeni kullanılabilir.
Son satır her zaman:  PYTHON=<ortamın python yolu>
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

VENV = Path(os.environ.get("STEMS_VENV", Path.home() / ".stems-venv"))
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

CHECK = (
    "import torch, audio_separator, soundfile, audioread;"
    "dev = 'cuda' if torch.cuda.is_available() else "
    "('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu');"
    "print(dev)"
)


def run(*cmd: str) -> None:
    print("  >", " ".join(str(c) for c in cmd[:6]), "..." if len(cmd) > 6 else "")
    subprocess.run([str(c) for c in cmd], check=True)


def pip(*args: str) -> None:
    run(PY, "-m", "pip", "install", "--disable-pip-version-check", "-q", *args)


def has_nvidia() -> bool:
    if IS_MAC or not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def device() -> str | None:
    if not PY.exists():
        return None
    r = subprocess.run([str(PY), "-c", CHECK], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def install() -> None:
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        sys.exit(f"Python 3.10-3.13 gerekli, bu {platform.python_version()}. Uygun bir Python ile çalıştırın.")
    print(f"Ortam kuruluyor: {VENV}  (ilk kurulum 2-4 GB indirir, birkaç dakika sürer)")
    run(sys.executable, "-m", "venv", VENV)
    run(PY, "-m", "pip", "install", "--disable-pip-version-check", "-q", "--upgrade", "pip")

    gpu = has_nvidia()
    pip("audio-separator[gpu]" if gpu else "audio-separator[cpu]", "audioread", "soundfile", "numpy", "setuptools<82")
    if gpu:
        # audio-separator PyPI'dan CPU'luk torch getiriyor; NVIDIA için üçünü birden CUDA'lı, uyumlu sürümle değiştir.
        print("NVIDIA ekran kartı bulundu, CUDA'lı PyTorch kuruluyor")
        pip("--force-reinstall", "--no-deps", "torch", "torchaudio", "torchvision", "--index-url", CUDA_INDEX)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows konsolunda Türkçe karakterler bozulmasın
    ap = argparse.ArgumentParser()
    ap.add_argument("--yeniden", action="store_true", help="Ortamı silip baştan kur")
    args = ap.parse_args()

    if args.yeniden and VENV.exists():
        shutil.rmtree(VENV)

    dev = device()
    if dev is None:
        install()
        dev = device()
        if dev is None:
            sys.exit("Kurulum doğrulanamadı: ortam içinde audio_separator/torch import edilemiyor.")
    else:
        print(f"Ortam zaten kurulu: {VENV}")

    print(f"Hesaplama aygıtı: {dev}" + ("  (ekran kartı yok, CPU ile 4-10 kat yavaş olur)" if dev == "cpu" else ""))
    if not shutil.which("ffmpeg"):
        hint = "winget install Gyan.FFmpeg" if IS_WIN else ("brew install ffmpeg" if IS_MAC else "sudo apt install ffmpeg")
        print(f"UYARI: ffmpeg bulunamadı. WAV dışı dosyalar (mp3, m4a...) için gerekli. Kurmak için: {hint}")
    print(f"PYTHON={PY}")


if __name__ == "__main__":
    main()
