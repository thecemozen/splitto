"""
Ayrılmış stem'lerden MIDI çıkarır (stems.py tarafından çağrılır, tek başına da çalışır).

1) Şarkı analizi: vuruş haritası (davuldan) ve ton (enstrümantal + vokalden). Tempo sabit varsayılmaz;
   MIDI'ye vuruş vuruş tempo haritası yazılır, böylece DAW'ın grid'i şarkının gerçek vuruşlarına oturur.
2) Bas: pYIN perde takibi + vuruş başlangıçları. Basic Pitch kadar isabetli, daha az notayla ve oktav hatası
   yapmadan; iki perde arasında kalan sesler tona uyan notaya yuvarlanır.
3) Ana vokal: Basic Pitch, aynı anda tek nota kalacak şekilde (armonikler atılır).
4) Çok sesli stem'ler (arka vokal, gitar, piyano, synth): Basic Pitch; tona uymayan kısa ve zayıf notalar atılır.
5) Davul: DrumSep parçalarındaki vuruş anları -> General MIDI davul notaları; sızıntılar elenir.
6) Hepsi 1/16'lık grid'e oturtulur (quantize).
7) Sesin açıkça onaylamadığı notalar atılır (kanıt filtresi): yanlış nota kulak için kaçırılan notadan kötü.
(Yöntem ve eşik seçimleri gerçek bir şarkıda yan yana ölçülerek yapıldı: doğruluk ve kapsama.)

Çıktı: <stem klasörü>/MIDI/  ->  her stem için bir .mid + güvenilir izlerden 00_Tum_Enstrumanlar.mid
Kontrol: her MIDI, üretimden bağımsız bir yöntemle kendi stem'ine karşı ölçülür (doğruluk %, kapsama %);
doğruluğu %95'in altındaki izler hepsi-bir-arada dosyaya girmez.

Tek başına:  python midi_cikar.py "<şarkı> - Stems" [--ham]   (--ham: grid'e oturtma)
"""
import contextlib
import io
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SR = 22050
HOP = 512  # ~23 ms

# stem -> (iz adı, GM program, min Hz, max Hz, yöntem)
#   pyin    : tek sesli perde takibi
#   bp_mono : Basic Pitch, aynı anda tek nota
#   bp_poly : Basic Pitch, akorlar serbest
MELODIC = {
    "01_Ana_Vokal": ("Ana Vokal", 54, 75, 1100, "bp_mono"),
    "02_Arka_Vokal": ("Arka Vokal", 52, 80, 1100, "bp_poly"),
    "04_Bas": ("Bas", 33, 30, 330, "pyin"),
    "05_Gitar": ("Gitar", 27, 70, 1400, "bp_poly"),
    "06_Piyano": ("Piyano", 0, 27, 4200, "bp_poly"),
    "07_Diger_Synth_Pad": ("Synth Pad", 89, 40, 4200, "bp_poly"),
}
# Ana vokal ayrılmadıysa (--mod vokal) tam vokalden melodi çıkar.
FALLBACK_VOCAL = ("00_Vokal_Tam", ("Vokal", 54, 75, 1100, "bp_mono"))

# davul parçası -> General MIDI davul notası
DRUM_NOTES = {
    "08_Davul_Kick": 36,
    "09_Davul_Snare": 38,
    "10_Davul_Tom": 45,
    "11_Davul_HiHat": 42,
    "12_Davul_Ride": 51,
    "13_Davul_Crash": 49,
}

# Davul parçasının en güçlü vuruşlarına göre en düşük kabul edilen seviye (0-1)
DRUM_FLOOR = {"08_Davul_Kick": 0.2, "09_Davul_Snare": 0.25, "11_Davul_HiHat": 0.6, "13_Davul_Crash": 0.45, "diger": 0.45}

# Bağımsız kontrolde doğruluğu bunun altında kalan izler "hepsi bir arada" dosyaya girmez
# (ayrı dosya olarak verilir, raporda işaretlenir). Kulak için yanlış nota kaçırılan notadan kötü.
RELIABLE = 0.95

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler ton profilleri
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class Song:
    beats: np.ndarray  # vuruş zamanları (sn); [0] = 0, sonuncusu şarkı bitiminden sonra
    tonic: int
    minor: bool
    quantize: bool

    def __post_init__(self):
        # her vuruşu 4'e böl: 1/16'lık grid, şarkının kendi (değişken) temposunu izler
        b = self.beats
        self.grid = np.concatenate([b[i] + (b[i + 1] - b[i]) * np.arange(4) / 4 for i in range(len(b) - 1)] + [b[-1:]])

    @property
    def tempos(self) -> np.ndarray:  # pickup vuruşu hariç her vuruşun temposu
        return 60.0 / np.diff(self.beats[1:])

    @property
    def bpm(self) -> float:
        return float(np.median(self.tempos))

    @property
    def step(self) -> float:  # tipik 1/16 nota süresi
        return 60.0 / self.bpm / 4

    @property
    def scale(self) -> set:
        steps = [0, 2, 3, 5, 7, 8, 10] if self.minor else [0, 2, 4, 5, 7, 9, 11]
        return {(self.tonic + s) % 12 for s in steps}

    @property
    def key_name(self) -> str:
        return f"{KEY_NAMES[self.tonic]} {'minör' if self.minor else 'majör'}"

    def q(self, t: float) -> float:
        if not self.quantize:
            return t
        i = int(np.clip(np.searchsorted(self.grid, t), 1, len(self.grid) - 1))
        return float(self.grid[i] if self.grid[i] - t < t - self.grid[i - 1] else self.grid[i - 1])


def _quiet():
    """Basic Pitch / onnxruntime'ın gürültülü uyarılarını ve print'lerini sustur."""
    logging.disable(logging.WARNING)
    return contextlib.redirect_stdout(io.StringIO())


def _load(path: Path, sr: int = SR):
    import librosa

    return librosa.load(str(path), sr=sr, mono=True)[0]


# ---------------------------------------------------------------- şarkı analizi

def beat_map(dest: Path) -> np.ndarray:
    """Şarkının vuruşlarını bulur. Tempo sabit varsayılmaz: AI üretimi ve canlı kayıtlarda tempo şarkı boyunca
    kayabiliyor (ör. 98.6 -> 99.7 BPM) ve sabit tempolu MIDI sonlara doğru vuruşlardan uzaklaşıyor."""
    import librosa
    import scipy.signal

    src = next((dest / n for n in ("03_Davul.wav", "00_Enstrumantal.wav", "00_Vokal_Tam.wav")
                if (dest / n).exists()), None)
    y = _load(src) if src is not None else np.zeros(SR)
    duration = len(y) / SR
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=hop, units="time", tightness=400)

    if len(beats) < 8:
        beats = np.arange(0, duration + 1.0, 0.5)  # ritim bulunamadı: 120 BPM
    else:
        bpm = 60.0 / np.median(np.diff(beats))
        if bpm < 70 or bpm > 180:  # yarım/çift tempo yakalandıysa makul aralığa getir
            factor = 2 if bpm < 70 else 0.5
            beats = (np.interp(np.arange(0, len(beats) - 0.5, 1 / factor), np.arange(len(beats)), beats)
                     if factor == 2 else beats[::2])
        win = min(17, len(beats) - (1 - len(beats) % 2))
        beats = scipy.signal.savgol_filter(beats, win, 2)  # vuruş titremesini yumuşat, yavaş tempo değişimini koru

    # baştan ve sondan şarkıyı kapsayacak şekilde uzat
    p0, p1 = beats[1] - beats[0], beats[-1] - beats[-2]
    head = beats[0] - p0 * np.arange(int(beats[0] / p0) + 1, 0, -1)
    tail = beats[-1] + p1 * np.arange(1, int((duration - beats[-1]) / p1) + 3)
    beats = np.concatenate([head, beats, tail])
    beats = beats[beats > 1e-3]
    # MIDI 0. saniyede bir vuruşla başlamalı: ilk vuruş çok yakınsa onu 0'a çek, değilse araya bir "pickup" koy
    if beats[0] < 0.5 * p0:
        beats = beats[1:]
    return np.concatenate([[0.0], beats])


def analyze(dest: Path, quantize: bool) -> Song:
    import librosa

    beats = beat_map(dest)

    chroma = np.zeros(12)
    for name in ("00_Enstrumantal.wav", "00_Vokal_Tam.wav"):
        if (dest / name).exists():
            y = _load(dest / name)
            chroma += librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP * 4).mean(axis=1)
    best = (-2.0, 0, False)
    for tonic in range(12):
        for minor, prof in ((False, MAJOR), (True, MINOR)):
            r = np.corrcoef(chroma, np.roll(prof, tonic))[0, 1]
            if r > best[0]:
                best = (r, tonic, minor)
    return Song(beats=beats, tonic=best[1], minor=best[2], quantize=quantize)


# ---------------------------------------------------------------- notaları temizleme

def tidy(notes: list, song: Song) -> list:
    """(start, end, pitch, vel) listesini grid'e oturtur; çok kısaları atar, aynı perdede çakışanları keser.
    Tekrarlanan notalar (ör. bas sekizlikleri) ayrı kalır."""
    out = []
    for s, e, p, v in notes:
        if e - s < song.step * 0.5:
            continue  # 1/32'den kısa: büyük ihtimalle hatalı nota
        qs, qe = song.q(s), song.q(e)
        if qe - qs < song.step:
            qe = qs + song.step
        out.append([qs, qe, int(p), int(v)])
    out.sort(key=lambda n: (n[2], n[0], -n[3]))
    kept = []
    for n in out:
        last = kept[-1] if kept else None
        if last and last[2] == n[2]:
            if n[0] == last[0]:
                continue  # aynı grid noktasına düşen ikinci nota
            if n[0] < last[1]:
                last[1] = n[0]  # öncekini yenisinin başında kes
        kept.append(n)
    kept.sort(key=lambda n: (n[0], n[2]))
    return kept


def join_fragments(notes: list, song: Song) -> list:
    """Tek sesli çizgide perde titremesinin böldüğü notaları birleştirir (A-B-A, B çok kısa ise -> A).
    Vuruşla bölünmüş tekrarlanan notalar arasında boşluk yoktur, onlara dokunulmaz."""
    notes = [n for n in notes if n[1] - n[0] >= song.step * 0.5]
    merged = []
    for n in sorted(notes):
        last = merged[-1] if merged else None
        if last and last[2] == n[2] and 0 < n[0] - last[1] <= song.step * 0.5:
            merged[-1] = (last[0], n[1], last[2], max(last[3], n[3]))
        else:
            merged.append(n)
    return merged


# ---------------------------------------------------------------- tek sesli (pYIN)

def mono_notes(wav: Path, fmin: float, fmax: float, song: Song, min_prob: float = 0.0) -> list:
    import librosa
    import scipy.signal

    y = _load(wav)
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=4096 if fmin < 60 else 2048,
                                    hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, prob, rms = f0[:n], voiced[:n], prob[:n], rms[:n]
    loud = rms > np.percentile(rms, 95) * 10 ** (-35 / 20)  # en güçlü kısımların 35 dB altı: sessizlik
    # pYIN'in kendi "ses var" kararı (HMM) yeterince iyi; olasılık eşiği eklemek notaların yarısını kaçırıyor.
    on = voiced & loud & np.isfinite(f0) & (prob >= min_prob)
    if not on.any():
        return []

    midi = np.full(n, np.nan)
    midi[on] = librosa.hz_to_midi(f0[on])
    # Boşlukları komşu değerlerle doldurup vibrato ve kaymaları yumuşat (~120 ms medyan);
    # boşluğa sıfır koymak kenarlarda sahte çok pes notalar üretiyordu.
    idx = np.arange(n)
    filled = np.interp(idx, idx[on], midi[on])
    sm = scipy.signal.medfilt(filled, 5)

    scale = song.scale
    pitch = np.full(n, -1, dtype=int)
    for i in np.where(on)[0]:
        m = sm[i]
        lo, hi = int(np.floor(m)), int(np.ceil(m))
        frac = m - lo
        if 0.3 < frac < 0.7 and ((lo % 12 in scale) != (hi % 12 in scale)):
            pitch[i] = lo if lo % 12 in scale else hi  # iki perde arasında: tona uyanı seç
        else:
            pitch[i] = int(round(m))
    # tek-iki karelik perde zıplamalarını temizle (sessiz kareler komşu perdeyle doldurulup sonra geri silinir)
    pitch[~on] = np.round(sm[~on]).astype(int)
    pitch = scipy.signal.medfilt(pitch, 5).astype(int)
    pitch[~on] = -1

    onsets = set(librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP, backtrack=True, units="frames").tolist())
    ref = np.percentile(rms[on], 95) if on.any() else 1.0
    times = librosa.frames_to_time(np.arange(n + 1), sr=SR, hop_length=HOP)

    notes, start = [], None
    for i in range(n + 1):
        cur = pitch[i] if i < n else -1
        boundary = i == n or cur != (pitch[start] if start is not None else -2) or (i in onsets and start is not None and i - start > 2)
        if start is not None and boundary:
            seg = slice(start, i)
            vel = int(np.clip(45 + 80 * min(np.mean(rms[seg]) / ref, 1.0), 1, 127))
            notes.append((times[start], times[i], int(pitch[start]), vel))
            start = None
        if start is None and i < n and cur >= 0:
            start = i
    return join_fragments(notes, song)


# ---------------------------------------------------------------- Basic Pitch

def monophonic(events: list) -> list:
    """Çakışan notalardan en güçlüsünü tutar (zayıf olan genelde bir armoniktir); öncekini yenisinde keser."""
    kept = []
    for ev in sorted(events, key=lambda e: (e[0], -e[3])):
        start, end, pitch, amp = ev[:4]
        if kept and start < kept[-1][1]:
            prev = kept[-1]
            if amp <= prev[3]:
                continue
            kept[-1] = (prev[0], start, prev[2], prev[3])
        kept.append((start, end, pitch, amp))
    return kept


def mono_bp_notes(wav: Path, fmin: float, fmax: float, model) -> list:
    """Vokal melodisi: varsayılan eşikler + aynı anda tek nota. (Eşikleri gevşetmek kaçan notayı azaltıyor ama
    nefes ve kaymalardan yanlış notalar ekliyor; kulak için yanlış nota daha kötü.)"""
    from basic_pitch.inference import predict

    with _quiet():
        _, _, events = predict(str(wav), model, minimum_note_length=120,
                               minimum_frequency=fmin, maximum_frequency=fmax)
    return [(s, e, int(p), int(np.clip(30 + a * 97, 1, 127))) for s, e, p, a in monophonic(events)]


def poly_notes(wav: Path, fmin: float, fmax: float, song: Song, model) -> list:
    from basic_pitch.inference import predict

    with _quiet():
        _, _, events = predict(
            str(wav), model, minimum_note_length=max(90.0, song.step * 600),
            minimum_frequency=fmin, maximum_frequency=fmax,
        )
    if not events:
        return []
    amps = np.array([e[3] for e in events])
    floor = 0.15
    scale = song.scale
    notes = []
    for start, end, pitch, amp, *_ in events:
        if amp < floor:
            continue  # en zayıf notalar: çoğunlukla armonik ya da başka enstrümandan sızıntı
        if pitch % 12 not in scale and (end - start) < 2 * song.step and amp < np.median(amps):
            continue  # tona uymayan, kısa ve zayıf: büyük ihtimalle hata
        notes.append((start, end, int(pitch), int(np.clip(30 + amp * 97, 1, 127))))
    return notes


# ---------------------------------------------------------------- kanıt filtresi
# Kulak için yanlış bir nota, kaçırılan bir notadan çok daha kötü. Sesin açıkça onaylamadığı notalar atılır.
# Filtre CQT (tek frekans kutusu) kullanır; kontrol (melodic_scores) ise farklı bir yöntemle, STFT harmonik toplamıyla ölçer.

def _cqt_db(wav: Path) -> np.ndarray:
    import librosa

    y = _load(wav)
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(24), n_bins=85, bins_per_octave=12))
    db = librosa.amplitude_to_db(C, ref=np.max)
    return db - db.max(axis=0, keepdims=True)


def evidence_filter(wav: Path, notes: list, mono: bool, rank: int = 3, floor_db: float = -15, frac: float = 0.6,
                    D: np.ndarray | None = None) -> list:
    D = _cqt_db(wav) if D is None else D
    n = D.shape[1]
    kept = []
    for note in notes:
        s, e, p = note[0], note[1], note[2]
        a = int(np.clip(s * SR / HOP, 0, n - 1))
        b = int(np.clip(e * SR / HOP, a + 1, n))
        i = p - 24
        if not 0 < i < D.shape[0] - 1:
            continue
        seg = D[:, a:b]
        if mono:
            top = seg.argmax(axis=0)
            # baskın perde bu nota, ya da bir oktav üstü (pes seslerde 2. harmonik temelden güçlü olabiliyor)
            hit = (np.abs(top - i) <= 1) | ((np.abs(top - i - 12) <= 1) & (seg[i] > -10))
        else:
            local = (seg[i] >= np.maximum(seg[i - 1], seg[i + 1])) & (seg[i] > floor_db)
            peaks = (seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:])
            stronger = np.sum(peaks & (seg[1:-1] > seg[i][None, :]), axis=0)
            hit = local & (stronger < rank)
        if np.mean(hit) >= frac:
            kept.append(note)
    return kept


def drop_glitches(notes: list, song: Song) -> list:
    """Tek seste: kısa olup hem önceki hem sonraki notadan çok uzak (> 7 yarım ses) duran notalar kaymadır, atılır."""
    notes = sorted(notes)
    out = []
    for k, nt in enumerate(notes):
        short = nt[1] - nt[0] < 2 * song.step
        prev = notes[k - 1][2] if k > 0 else None
        nxt = notes[k + 1][2] if k + 1 < len(notes) else None
        far = all(x is None or abs(nt[2] - x) > 7 for x in (prev, nxt)) and (prev is not None or nxt is not None)
        if not (short and far):
            out.append(nt)
    return out


# ---------------------------------------------------------------- davul

def drum_notes(dest: Path, song: Song) -> list:
    """Her davul parçasında vuruş anlarını bulur; diğer parçalardan sızan sesleri seviye eşiğiyle eler."""
    import librosa

    hits = {}  # stem -> [(zaman sn, göreli seviye 0-1)]
    for stem in DRUM_NOTES:
        wav = dest / f"{stem}.wav"
        if not wav.exists():
            continue
        y = _load(wav, 44100)
        hop = 256
        env = librosa.onset.onset_strength(y=y, sr=44100, hop_length=hop)
        onsets = librosa.onset.onset_detect(
            onset_envelope=env, sr=44100, hop_length=hop, units="samples", backtrack=False, delta=0.08, wait=4
        )
        if len(onsets) == 0:
            continue
        win = int(0.03 * 44100)  # vuruştan sonraki 30 ms'nin tepe seviyesi
        peaks = np.array([np.max(np.abs(y[o:o + win])) if o < len(y) else 0.0 for o in onsets])
        ref = np.percentile(peaks, 95) or 1e-9
        # Parçanın en güçlü vuruşlarına göre zayıf olanlar büyük ihtimalle başka davuldan sızıntı.
        # Zil ve tom'lar DrumSep'te en çok sızıntı alan kanallar, eşikleri daha yüksek.
        floor = DRUM_FLOOR.get(stem, DRUM_FLOOR["diger"])
        hits[stem] = [(o / 44100, min(p / ref, 1.0)) for o, p in zip(onsets, peaks) if p / ref >= floor]

    def near(t, others, tol=0.03):
        return len(others) and np.min(np.abs(others - t)) <= tol

    # Kick/snare vuruşu diğer kanallara sızabiliyor: aynı anda gelen, çok güçlü olmayan vuruşları at.
    strong = np.array(sorted(t for s in ("08_Davul_Kick", "09_Davul_Snare") for t, _ in hits.get(s, [])))
    for stem, keep_above in (("10_Davul_Tom", 0.7), ("12_Davul_Ride", 0.7), ("13_Davul_Crash", 0.7), ("11_Davul_HiHat", 0.5)):
        if stem in hits:
            hits[stem] = [(t, r) for t, r in hits[stem] if r >= keep_above or not near(t, strong)]
    # Crash bir vuruştan uzun çınlar; bir vuruştan sık gelen crash'ler hi-hat sızıntısıdır: en güçlüsü kalır.
    if "13_Davul_Crash" in hits:
        beat = 4 * song.step
        kept = []
        for t, r in sorted(hits["13_Davul_Crash"], key=lambda x: -x[1]):
            if all(abs(t - k) >= beat for k, _ in kept):
                kept.append((t, r))
        hits["13_Davul_Crash"] = sorted(kept)

    slots = {}  # (perde, grid zamanı) -> en güçlü vuruş; grid'e oturunca çift vuruşlar tekleşir
    for stem, items in hits.items():
        for t, rel in items:
            key = (DRUM_NOTES[stem], round(song.q(t), 4))
            vel = int(np.clip(40 + 87 * rel, 1, 127))
            slots[key] = max(slots.get(key, 0), vel)
    return sorted((t, t + 0.1, p, v) for (p, t), v in slots.items())


# ---------------------------------------------------------------- yazma

def _instrument(name: str, program: int, notes: list, drum: bool = False):
    import pretty_midi

    inst = pretty_midi.Instrument(program=program, is_drum=drum, name=name)
    for s, e, p, v in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(s), end=float(e)))
    return inst


def _midi_file(song: Song, instruments: list):
    """Vuruş vuruş tempo haritalı MIDI: her vuruş tam bir çeyrek nota olur, DAW grid'i şarkının vuruşlarını izler."""
    import pretty_midi

    res = 480
    pm = pretty_midi.PrettyMIDI(resolution=res)
    # pretty_midi tempo haritasını _tick_scales'ten yazar: (tick, saniye/tick) çiftleri
    pm._tick_scales = [(i * res, float(d) / res) for i, d in enumerate(np.diff(song.beats))]
    pm._update_tick_to_time(len(song.beats) * res)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    pm.key_signature_changes.append(pretty_midi.KeySignature(song.tonic + (12 if song.minor else 0), 0))
    pm.instruments.extend(instruments)
    return pm


def extract(dest: Path, quantize: bool = True) -> list[Path]:
    """dest içindeki stem'lerden MIDI üretir, yazılan .mid dosyalarını döndürür."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    out = dest / "MIDI"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.mid"):
        old.unlink()

    song = analyze(dest, quantize)
    t = song.tempos
    tempo_text = (f"{song.bpm:.1f} BPM (şarkı boyunca {np.percentile(t, 5):.1f}-{np.percentile(t, 95):.1f} arası değişiyor,"
                  f" MIDI'ye vuruş vuruş tempo haritası yazıldı)" if np.ptp(np.percentile(t, [5, 95])) > 0.3
                  else f"{song.bpm:.1f} BPM (sabit)")
    print(f"   tempo: {tempo_text}, ton: {song.key_name}")
    (out / "_bilgi.txt").write_text(
        f"Tempo: {tempo_text}\nTon: {song.key_name}\nGrid: {'1/16' if quantize else 'yok (ham)'}\n", encoding="utf-8")

    jobs = {s: v for s, v in MELODIC.items() if (dest / f"{s}.wav").exists()}
    if "01_Ana_Vokal" not in jobs and (dest / f"{FALLBACK_VOCAL[0]}.wav").exists():
        jobs[FALLBACK_VOCAL[0]] = FALLBACK_VOCAL[1]

    model = None
    if any(v[4] != "pyin" for v in jobs.values()):
        with _quiet():
            model = Model(ICASSP_2022_MODEL_PATH)
    tracks, scores = [], {}
    for stem, (name, program, fmin, fmax, method) in jobs.items():
        wav = dest / f"{stem}.wav"
        mono = method != "bp_poly"
        if method == "pyin":
            notes = mono_notes(wav, fmin, fmax, song)
        elif method == "bp_mono":
            notes = mono_bp_notes(wav, fmin, fmax, model)
        else:
            notes = poly_notes(wav, fmin, fmax, song, model)
        if mono:
            notes = drop_glitches(evidence_filter(wav, notes, True), song)
        else:
            # akorlarda Basic Pitch'in yanlış notası çok: sadece o anın en güçlü 2 perdesinden biri olan notalar kalır
            notes = evidence_filter(wav, notes, False, rank=2, floor_db=-12, frac=0.7)
        notes = tidy(notes, song)
        acc, cov = melodic_scores(wav, notes, mono)
        scores[stem] = {"dogruluk": acc, "kapsama": cov}
        tracks.append((stem, _instrument(name, program, notes)))

    drums = drum_notes(dest, song)
    if drums:
        # her davul parçası ayrı doğrulanır; güvenilmez parçalar (ör. sızıntıdan oluşan crash) davul izinden çıkarılır
        kept, dropped = [], []
        for stem, pitch in DRUM_NOTES.items():
            part = [n for n in drums if n[2] == pitch]
            if not part:
                continue
            acc = drum_scores(dest, part)
            scores[f"03_Davul/{stem[9:]}"] = {"dogruluk": acc, "vurus": len(part)}
            (kept if acc >= RELIABLE else dropped).extend(part)
        if dropped:
            print(f"   ! davul: güvenilmez parçalar çıkarıldı ({len(dropped)} vuruş)")
        if kept:
            scores["03_Davul"] = {"dogruluk": drum_scores(dest, kept)}
            tracks.append(("03_Davul", _instrument("Davul", 0, sorted(kept), drum=True)))

    tracks = sorted((t for t in tracks if t[1].notes), key=lambda t: t[0])
    written = []
    for stem, inst in tracks:
        path = out / f"{stem}.mid"
        _midi_file(song, [inst]).write(str(path))
        written.append(path)
        mark = "" if scores[stem]["dogruluk"] >= RELIABLE else "  (güvenilmez, hepsi-bir-arada dosyaya girmedi)"
        print(f"   + MIDI/{path.name} ({len(inst.notes)} nota, doğruluk %{scores[stem]['dogruluk'] * 100:.1f}){mark}")
    good = [inst for stem, inst in tracks if scores[stem]["dogruluk"] >= RELIABLE]
    if good:
        path = out / "00_Tum_Enstrumanlar.mid"
        _midi_file(song, good).write(str(path))
        written.append(path)
        print(f"   + MIDI/{path.name} ({len(good)} güvenilir iz)")
    (out / "_puanlar.json").write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
    return written


# ---------------------------------------------------------------- kontrol
# Not üretiminden bağımsız yöntemlerle ölçülür (STFT harmonik toplamı, bant enerjisi), böylece
# filtrenin kendi kendini "doğrulaması" engellenir.

def _salience(wav: Path):
    """Her kare ve her MIDI perdesi (24-108) için harmonik toplam (dB, karedeki en güçlü perdeye göre)."""
    import librosa

    y = _load(wav)
    S = np.abs(librosa.stft(y, n_fft=8192, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=8192)
    pitches = np.arange(24, 109)
    sal = np.zeros((len(pitches), S.shape[1]))
    for h, w in zip(range(1, 6), (1.0, 0.8, 0.6, 0.5, 0.4)):
        f = librosa.midi_to_hz(pitches) * h
        idx = np.clip(np.searchsorted(freqs, f), 0, len(freqs) - 1)
        sal += w * S[idx]
    db = librosa.amplitude_to_db(sal, ref=np.max)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    return db - db.max(axis=0, keepdims=True), rms


def melodic_scores(wav: Path, notes: list, mono: bool) -> tuple[float, float]:
    """(doğruluk, kapsama): doğrulanan nota oranı ve sesin MIDI'ye geçen kısmı.
    Tek seste nota o anın baskın perdesi olmalı (±1 yarım ses); çok seste en güçlü 4 tepe arasında olmalı."""
    sal, rms = _salience(wav)
    n = sal.shape[1]
    t2f = lambda t: int(np.clip(round(t * SR / HOP), 0, n - 1))  # noqa: E731
    ok = 0
    covered = np.zeros(n, dtype=bool)
    for s, e, p, _ in notes:
        a, b = t2f(s), max(t2f(e), t2f(s) + 1)
        covered[a:b] = True
        i = p - 24
        if not 0 < i < sal.shape[0] - 1:
            continue
        seg = sal[:, a:b]
        if mono:
            hit = np.abs(seg.argmax(axis=0) - i) <= 1
        else:
            local = (seg[i] >= np.maximum(seg[i - 1], seg[i + 1])) & (seg[i] > -12)
            # i'den güçlü olan yerel tepe sayısı
            peaks = (seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:])
            stronger = np.sum(peaks & (seg[1:-1] > seg[i][None, :]), axis=0)
            hit = local & (stronger < 4)
        ok += np.mean(hit) >= 0.5
    loud = rms[:n] > np.percentile(rms[:n], 60)
    return ok / max(len(notes), 1), float(np.mean(covered[:len(loud)][loud]))


DRUM_BANDS = {36: (30, 150), 38: (1500, 5000), 45: (80, 400), 42: (7000, 16000), 51: (4000, 12000), 49: (4000, 16000)}


def drum_scores(dest: Path, notes: list) -> float:
    """Davul doğruluğu: her vuruş anında, tüm davul stem'inde o davulun frekans bandında enerji sıçraması var mı."""
    import librosa

    src = dest / "03_Davul.wav"
    if not src.exists() or not notes:
        return 0.0
    sr, hop = 44100, 256
    y = _load(src, sr)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = {p: 10 * np.log10(S[(freqs >= lo) & (freqs < hi)].sum(0) + 1e-12) for p, (lo, hi) in DRUM_BANDS.items()}
    n = S.shape[1]
    ok = 0
    for s, _, p, _ in notes:
        f = int(s * sr / hop)
        e = bands.get(p)
        if e is None or f < 4 or f + 6 > n:
            continue
        if p == 49:
            # crash uzun çınlar: vuruştan 100-400 ms sonra tiz bant, vuruş öncesinden belirgin yüksek kalmalı
            # (anlık sıçrama ölçülemez, sürekli çalan hi-hat aynı bantta)
            post = e[f + 17:f + 70].mean() if f + 70 < n else -np.inf
            pre = e[max(f - 26, 0):max(f - 3, 1)].mean()
            ok += post - pre > 2.0
            continue
        # vuruştan sonraki 30 ms ile önceki 40 ms (grid kaymasına pay bırakarak ±2 kare tara);
        # tom ve ride, snare/hi-hat ile aynı bantları paylaştığı için daha büyük sıçrama ister
        need = 6.0 if p in (45, 51) else 4.0
        jumps = [e[g:g + 5].max() - e[g - 7:g - 1].mean() for g in range(max(f - 2, 7), min(f + 3, n - 5))]
        ok += max(jumps, default=0) > need
    return ok / len(notes)


def grid_offset_ms(dest: Path, mid: Path) -> float | None:
    """Davuldaki gerçek vuruşların MIDI'nin 1/16 grid çizgilerine medyan uzaklığı (ms). Ritim oturuyor mu?"""
    import librosa
    import pretty_midi

    src = dest / "03_Davul.wav"
    if not src.exists():
        return None
    beats = pretty_midi.PrettyMIDI(str(mid)).get_beats()
    if len(beats) < 4:
        return None
    grid = np.concatenate([beats[i] + (beats[i + 1] - beats[i]) * np.arange(4) / 4 for i in range(len(beats) - 1)])
    y = _load(src)
    hits = librosa.onset.onset_detect(y=y, sr=SR, hop_length=128, units="time")
    hits = hits[hits > beats[1]]  # pickup vuruşunu sayma
    if len(hits) == 0:
        return None
    idx = np.clip(np.searchsorted(grid, hits), 1, len(grid) - 1)
    d = np.minimum(np.abs(grid[idx] - hits), np.abs(hits - grid[idx - 1]))
    return float(np.median(d) * 1000)


def verify(dest: Path, duration: float | None) -> tuple[list[str], bool]:
    """MIDI dosyalarını okur: nota var mı, süre aşımı, bas perdesi, doğruluk/kapsama, ritmin grid'e oturması."""
    import pretty_midi

    lines, ok = [], True
    files = sorted((dest / "MIDI").glob("*.mid")) if (dest / "MIDI").exists() else []
    if not files:
        return ["MIDI: hiç dosya yok"], False
    info = dest / "MIDI" / "_bilgi.txt"
    if info.exists():
        lines.append("\n" + info.read_text(encoding="utf-8").strip())
    off = grid_offset_ms(dest, files[0])
    if off is not None:
        verdict = "iyi" if off <= 20 else "KAYIK (tempo haritası vuruşları tutmuyor)"
        lines.append(f"Davul vuruşlarının MIDI grid'ine uzaklığı: {off:.0f} ms ({verdict})")
        ok = ok and off <= 20
    lines.append("\nDoğruluk: MIDI'deki notaların sesçe doğrulanan oranı. Kapsama: stem'in sesli kısımlarının MIDI'ye geçen oranı.")
    lines.append(f"{'MIDI':<24} {'Nota':>6} {'Aralık':>9} {'Doğruluk':>9} {'Kapsama':>8}  Durum")
    mono_stems = {s for s, v in [*MELODIC.items(), FALLBACK_VOCAL] if v[4] != "bp_poly"}
    reliable, total = 0, 0
    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
        except Exception as e:
            lines.append(f"{f.stem:<24} okunamadı: {e}")
            ok = False
            continue
        notes = [(n.start, n.end, n.pitch, n.velocity) for i in pm.instruments for n in i.notes]
        if not notes:
            lines.append(f"{f.stem:<24} {0:>6}  SORUN: boş")
            ok = False
            continue
        lo, hi = min(n[2] for n in notes), max(n[2] for n in notes)
        problems = []
        if duration and max(n[1] for n in notes) > duration + 1.0:
            problems.append("şarkı süresini aşıyor")
        if f.stem == "04_Bas" and np.median([n[2] for n in notes]) > 60:
            problems.append("bas notaları çok tiz")
        # puanlar dosyalardan yeniden hesaplanır (çıkarma sırasındaki değerlere güvenilmez)
        acc, cov = None, None
        if f.stem == "03_Davul":
            acc = drum_scores(dest, notes)
        elif (dest / f"{f.stem}.wav").exists():
            acc, cov = melodic_scores(dest / f"{f.stem}.wav", notes, f.stem in mono_stems)
        if acc is not None:
            total += 1
            reliable += acc >= RELIABLE
        if problems:
            status = "SORUN: " + "; ".join(problems)
        elif acc is not None and acc < RELIABLE:
            status = "güvenilmez: hepsi-bir-arada dosyada yok"
        else:
            status = "tamam"
        ok = ok and not problems
        rng = f"{pretty_midi.note_number_to_name(lo)}-{pretty_midi.note_number_to_name(hi)}"
        a = f"%{acc * 100:.1f}" if acc is not None else "-"
        c = f"%{cov * 100:.0f}" if cov is not None else "-"
        lines.append(f"{f.stem:<24} {len(notes):>6} {rng:>9} {a:>9} {c:>8}  {status}")

    sc = dest / "MIDI" / "_puanlar.json"
    if sc.exists():
        parts = {k: v for k, v in json.loads(sc.read_text(encoding="utf-8")).items() if k.startswith("03_Davul/")}
        if parts:
            lines.append("Davul parçaları: " + ", ".join(
                f"{k[9:]} %{v['dogruluk'] * 100:.0f}" + ("" if v["dogruluk"] >= RELIABLE else " (çıkarıldı)")
                for k, v in parts.items()))
    lines.append(f"Güvenilir iz: {reliable}/{total} (doğruluk >= %{RELIABLE * 100:.0f})")
    return lines, ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = Path(sys.argv[1])
    extract(d, quantize="--ham" not in sys.argv)
    print("\n".join(verify(d, None)[0]))
