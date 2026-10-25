"""
Bir şarkıyı vokal ve enstrüman stem'lerine ayırıp WAV olarak kaydeder, sonra her çıktıyı kontrol eder.

Akış (her adım bir öncekinin çıktısını işler):
  1) BS-RoFormer          -> Vokal (tam) + Enstrümantal
  2) Mel-RoFormer Karaoke -> Vokal içinden Ana Vokal + Arka Vokal
  3) Demucs htdemucs_6s   -> Enstrümantal içinden Davul, Bas, Gitar, Piyano, Diğer (synth/pad/yaylı)
  4) MDX23C DrumSep       -> Davul içinden Kick, Snare, Tom, Hi-Hat, Ride, Crash
Son olarak: sessiz (şarkıda olmayan) stem'ler silinir, kalanlar kontrol edilip _kontrol.txt yazılır.

Kullanım:
  python stems.py şarkı.wav [başka.mp3 klasör/ ...] [--cikti KLASÖR] [--mod tam|hizli|vokal] [--sessizleri-tut]
    --mod tam    : 4 adımın hepsi (varsayılan)
    --mod hizli  : davul seti parçalanmaz (1-3)
    --mod vokal  : sadece vokal + enstrümantal (1)
Çıktı varsayılan olarak şarkının yanında "<şarkı adı> - Stems" klasörüne yazılır.
Çıkış kodu: 0 = her şey tamam, 1 = en az bir şarkıda hata/kontrol sorunu.
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_separator.separator import Separator

VENV = Path(os.environ.get("STEMS_VENV", Path.home() / ".stems-venv"))
MODEL_DIR = VENV / "models"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".mp4", ".webm"}

SILENT_DB = -50.0  # RMS bunun altındaysa stem "boş" sayılır (ör. şarkıda piyano yoksa -70 dB civarı çıkar)

MODEL_VOCALS = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MODEL_KARAOKE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
MODEL_INSTRUMENTS = "htdemucs_6s.yaml"
MODEL_DRUMS = "MDX23C-DrumSep-aufr33-jarredou.ckpt"

# Modelin verdiği stem adı (küçük harf) -> çıktı dosya adı. None = atılır.
NAMES_VOCALS = {"vocals": "00_Vokal_Tam", "instrumental": "00_Enstrumantal"}
NAMES_KARAOKE = {"vocals": "01_Ana_Vokal", "instrumental": "02_Arka_Vokal"}
NAMES_INSTRUMENTS = {
    "drums": "03_Davul",
    "bass": "04_Bas",
    "guitar": "05_Gitar",
    "piano": "06_Piyano",
    "other": "07_Diger_Synth_Pad",
    "vocals": None,  # enstrümantalde kalan vokal artığı
}
NAMES_DRUMS = {
    "kick": "08_Davul_Kick",
    "snare": "09_Davul_Snare",
    "toms": "10_Davul_Tom",
    "hh": "11_Davul_HiHat",
    "ride": "12_Davul_Ride",
    "crash": "13_Davul_Crash",
}


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-10))


def stem_of(path: Path) -> str:
    """audio-separator dosya adındaki '(Stem)' kısmını döndürür."""
    found = re.findall(r"\(([^)]+)\)", path.stem)
    return found[-1].strip().lower() if found else ""


def run_model(sep: Separator, model: str, src: Path, work: Path, names: dict, dest: Path) -> dict:
    """Modeli çalıştırır, çıktıları dest'e istenen adlarla taşır. {stem: yol} döndürür."""
    sep.output_dir = str(work)
    sep.load_model(model_filename=model)
    produced = {}
    for f in sep.separate(str(src)):
        f = Path(f) if Path(f).is_absolute() else work / f
        key = stem_of(f)
        if key not in names:
            print(f"   ! beklenmeyen stem '{key}' ({f.name}), atlandı")
            continue
        if names[key] is None:
            f.unlink(missing_ok=True)
            continue
        target = dest / f"{names[key]}.wav"
        shutil.move(str(f), target)
        produced[key] = target
        print(f"   + {target.name}")
    return produced


def verify(song: Path, dest: Path, keep_silent: bool) -> bool:
    """Her stem'i kontrol eder, boşları siler, raporu yazar. Sorun yoksa True."""
    try:
        ref_info = sf.info(str(song))
        ref_sr, ref_dur = ref_info.samplerate, ref_info.duration
    except Exception:
        ref_sr = ref_dur = None  # soundfile okuyamadıysa (ör. m4a) karşılaştırmayı atla

    lines, ok = [], True
    header = f"{'Dosya':<24} {'Süre':>7} {'Hz':>6} {'RMS dB':>7} {'Tepe dB':>7}  Durum"
    lines.append(header)
    for wav in sorted(dest.glob("*.wav")):
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        dur = len(data) / sr
        rms, peak = db(float(np.sqrt(np.mean(data ** 2)))), db(float(np.max(np.abs(data))))
        problems = []
        if ref_sr and sr != ref_sr:
            problems.append(f"örnekleme hızı {sr} != {ref_sr}")
        if ref_dur and abs(dur - ref_dur) > 0.05:
            problems.append(f"süre {dur:.2f}s != {ref_dur:.2f}s")
        if peak >= -0.01:
            problems.append("clipping (tepe 0 dBFS)")

        if rms < SILENT_DB and not keep_silent:
            status = "boş -> silindi"
            wav.unlink()
        elif problems:
            status = "SORUN: " + "; ".join(problems)
            ok = False
        else:
            status = "tamam" + (" (boş)" if rms < SILENT_DB else "")
        lines.append(f"{wav.stem:<24} {dur:>6.1f}s {sr:>6} {rms:>7.1f} {peak:>7.1f}  {status}")

    # Vokal + enstrümantal toplamı orijinali vermeli; fark ne kadar düşükse ayrım o kadar kayıpsız.
    voc, ins = dest / "00_Vokal_Tam.wav", dest / "00_Enstrumantal.wav"
    try:
        orig, _ = sf.read(str(song), dtype="float32", always_2d=True)
        parts = [sf.read(str(p), dtype="float32", always_2d=True)[0] for p in (voc, ins) if p.exists()]
        if parts:
            n = min([len(orig)] + [len(p) for p in parts])
            recon = sum(p[:n] for p in parts)
            diff = db(float(np.sqrt(np.mean((orig[:n] - recon) ** 2)))) - db(float(np.sqrt(np.mean(orig[:n] ** 2))))
            verdict = "iyi" if diff < -20 else "YÜKSEK (ayrım kayıplı)"
            lines.append(f"\nVokal+Enstrümantal -> orijinal farkı: {diff:.1f} dB ({verdict})")
            ok = ok and diff < -20
    except Exception as e:
        lines.append(f"\nYeniden birleştirme kontrolü atlandı: {e}")

    remaining = len(list(dest.glob("*.wav")))
    if remaining == 0:
        lines.append("HATA: hiç stem kalmadı")
        ok = False
    lines.append(f"Kalan WAV: {remaining}  |  Genel sonuç: {'TAMAM' if ok else 'SORUN VAR'}")
    report = "\n".join(lines)
    (dest / "_kontrol.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    return ok


def process(song: Path, sep: Separator, mode: str, out_root: Path | None, keep_silent: bool) -> bool:
    dest = (out_root or song.parent) / f"{song.stem} - Stems"
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.wav"):  # yeniden çalıştırmada eski sonuçlar karışmasın
        old.unlink()
    t0 = time.time()
    print(f"\n=== {song.name} -> {dest}")

    # Stem'ler orijinalle aynı örnekleme hızında çıksın (DAW'da kaymasın); okunamazsa 44.1 kHz.
    try:
        sep.sample_rate = sf.info(str(song)).samplerate
    except Exception:
        sep.sample_rate = 44100

    steps = {"vokal": 1, "hizli": 3, "tam": 4}[mode]
    # Kısa adlı geçici klasör: model çıktılarının uzun dosya adları Windows yol sınırına takılmasın.
    with tempfile.TemporaryDirectory(prefix="stems_") as tmp:
        work = Path(tmp)
        src = work / f"in{song.suffix.lower()}"
        shutil.copy2(song, src)

        print(f"[1/{steps}] Vokal / enstrümantal ayrılıyor")
        base = run_model(sep, MODEL_VOCALS, src, work, NAMES_VOCALS, dest)

        if steps >= 3:
            print(f"[2/{steps}] Ana vokal / arka vokal ayrılıyor")
            run_model(sep, MODEL_KARAOKE, base["vocals"], work, NAMES_KARAOKE, dest)

            print(f"[3/{steps}] Enstrümanlar ayrılıyor (davul, bas, gitar, piyano, diğer)")
            inst = run_model(sep, MODEL_INSTRUMENTS, base["instrumental"], work, NAMES_INSTRUMENTS, dest)

            if steps == 4 and "drums" in inst:
                print(f"[4/{steps}] Davul seti parçalanıyor (kick, snare, tom, hi-hat, ride, crash)")
                run_model(sep, MODEL_DRUMS, inst["drums"], work, NAMES_DRUMS, dest)

    print(f"--- Ayırma bitti ({time.time() - t0:.0f} sn). Kontrol:")
    return verify(song, dest, keep_silent)


def collect(inputs: list[str]) -> list[Path]:
    songs = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            songs += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXT)
        elif p.is_file():
            songs.append(p)
        else:
            print(f"Bulunamadı: {p}")
    return songs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Şarkıyı vokal ve enstrüman stem'lerine ayırıp WAV kaydeder.")
    ap.add_argument("girdi", nargs="+", help="Ses dosyaları veya klasörler")
    ap.add_argument("--cikti", help="Çıktı ana klasörü (varsayılan: şarkının yanı)")
    ap.add_argument("--mod", choices=["tam", "hizli", "vokal"], default="tam")
    ap.add_argument("--sessizleri-tut", action="store_true", help="Boş stem'leri silme")
    args = ap.parse_args()

    songs = collect(args.girdi)
    if not songs:
        sys.exit("İşlenecek ses dosyası bulunamadı.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.cikti) if args.cikti else None
    # normalization_threshold=1.0: stem'ler yeniden ölçeklenmesin, DAW'da toplamları orijinali versin.
    sep = Separator(log_level=30, model_file_dir=str(MODEL_DIR), output_format="WAV", normalization_threshold=1.0)

    results = {}
    for song in songs:
        try:
            results[song.name] = process(song, sep, args.mod, out_root, args.sessizleri_tut)
        except Exception as e:  # bir şarkı patlarsa diğerlerine devam et
            print(f"HATA ({song.name}): {e}")
            results[song.name] = False

    print("\n=== ÖZET")
    for name, ok in results.items():
        print(f"{'TAMAM' if ok else 'SORUN'}  {name}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
