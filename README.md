# Splitto — Vocal Stems Splitter

Şarkıları kendi bilgisayarında, ücretsiz olarak vokal ve enstrüman katmanlarına (stem) ayıran bir **Claude skill'i**.
Suno'nun "Extract Stems" özelliği, UVR veya LALAL.AI gibi, ama tamamen yerel çalışır.

Claude'a sadece *"şu şarkıyı stem'lerine ayır"* demen yeterli. Gerisini skill halleder: kurulum, ayırma, boş
stem'leri silme ve her çıktıyı kontrol etme.

## Ne çıkarır?

| Dosya | İçerik |
|---|---|
| `00_Vokal_Tam.wav` | Bütün vokaller |
| `00_Enstrumantal.wav` | Vokalsiz altyapı |
| `01_Ana_Vokal.wav` | Ana (lead) vokal |
| `02_Arka_Vokal.wav` | Arka vokaller / armoniler |
| `03_Davul.wav` | Davulun tamamı |
| `04_Bas.wav` | Bas |
| `05_Gitar.wav` | Gitar |
| `06_Piyano.wav` | Piyano / tuşlu |
| `07_Diger_Synth_Pad.wav` | Synth, pad, yaylı ve kalan her şey |
| `08`–`13_Davul_*.wav` | Kick, Snare, Tom, Hi-Hat, Ride, Crash |

- Çıktılar orijinalle **aynı örnekleme hızında ve uzunlukta** olur. DAW'a sürükleyince birebir hizalanırlar.
- Şarkıda olmayan enstrümanların boş dosyaları **otomatik silinir**.
- Her iş sonunda tüm dosyalar **otomatik kontrol edilir**: süre, örnekleme hızı, seviye, clipping ve
  vokal+enstrümantal toplamının orijinali ne kadar verdiği. Rapor klasöre `_kontrol.txt` olarak yazılır.

## Kurulum

### Claude uygulaması / Claude.ai
1. [Releases](../../releases) sayfasından `splitto.skill` dosyasını indir.
2. Claude'da **Ayarlar → Capabilities → Skills** bölümünden dosyayı yükle.

### Claude Code
```bash
git clone https://github.com/thecemozen/splitto.git
```
Ardından `splitto/splitto` klasörünü `~/.claude/skills/` içine kopyala.

İlk kullanımda skill gerekli Python ortamını kendisi kurar (`~/.stems-venv`, 2-4 GB) ve modelleri indirir
(~1.5 GB). Bu kurulum sadece bir kez yapılır.

## Gereksinimler

- Python 3.10–3.13
- ffmpeg (WAV/FLAC dışındaki dosyalar için). Windows: `winget install Gyan.FFmpeg`, macOS: `brew install ffmpeg`
- NVIDIA ekran kartı önerilir. 6 GB'lık bir dizüstü kartında 3-4 dakikalık bir şarkı yaklaşık 10-12 dakika
  sürer. CPU ve Apple Silicon ile de çalışır, ama daha yavaştır.

## Claude olmadan kullanmak

```bash
python splitto/scripts/kurulum.py
~/.stems-venv/bin/python splitto/scripts/stems.py "şarkı.wav"              # macOS/Linux
%USERPROFILE%\.stems-venv\Scripts\python.exe splitto\scripts\stems.py "şarkı.wav"   # Windows
```

Seçenekler: `--mod tam|hizli|vokal`, `--cikti KLASÖR`, `--sessizleri-tut`.
Windows'ta ses dosyalarını [`windows/Splitto.bat`](windows/Splitto.bat) üzerine sürükleyip bırakmak da yeterli.

## Kullanılan modeller

[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) üzerinden:

| Adım | Model |
|---|---|
| Vokal / enstrümantal | BS-RoFormer (viperx, SDR 12.97) |
| Ana / arka vokal | Mel-RoFormer Karaoke (aufr33 & viperx) |
| Enstrümanlar | Demucs v4 `htdemucs_6s` (Meta) |
| Davul seti | MDX23C DrumSep (aufr33 & jarredou) |

Modellerin her birinin kendi lisansı vardır. Ayırdığın şarkıların haklarına sahip olduğundan emin ol.

## Lisans

Kod MIT lisansıyla dağıtılır, ayrıntılar [LICENSE](LICENSE) dosyasında.
