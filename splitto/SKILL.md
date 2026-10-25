---
name: splitto
description: Splitto — Vocal Stems Splitter. Bir şarkıyı yerel bilgisayarda, ücretsiz olarak vokal ve enstrüman stem'lerine ayırıp ayrı WAV dosyaları yapar (ana vokal, arka vokal, davul, bas, gitar, piyano, synth/pad ve davul setinin kick/snare/tom/hi-hat/ride/crash parçaları), her stem'in MIDI'sini çıkarır (melodi, bas, akor, davul), boş çıkan stem'leri siler ve her çıktıyı kontrol eder. Suno "Extract Stems and MIDI", UVR, Demucs, LALAL.AI benzeri. Kullanıcı "stem ayır", "stems çıkar", "vokali ayır", "enstrümanları ayır", "akapella / altyapı / karaoke çıkar", "MIDI çıkar", "notaları çıkar", "melodiyi MIDI yap", "split stems", "separate vocals", "extract stems", "audio to MIDI", "remove vocals", "instrumental version" dediğinde, ya da bir şarkının katmanlarını veya notalarını DAW'da kullanmak, remix/cover yapmak, vokalsiz altyapı istemek veya bir ses dosyasındaki enstrümanları ayrı ayrı almak istediğinde, açıkça "stem" demese bile bu skill'i kullan.
compatibility: Python 3.10-3.13, ffmpeg (WAV dışı girdiler için). NVIDIA ekran kartı önerilir; yoksa CPU/Apple Silicon ile yavaş çalışır. İlk kurulum 2-4 GB indirir.
---

# Splitto — Vocal Stems Splitter

Şarkıyı 4 model zinciriyle katmanlarına ayırır, hepsini orijinalle aynı örnekleme hızında WAV olarak yazar,
şarkıda olmayan enstrümanların boş dosyalarını siler, kalanlardan MIDI çıkarır ve sonra her dosyayı kontrol eder.

| Adım | Model | Çıktı |
|---|---|---|
| 1 | BS-RoFormer | `00_Vokal_Tam`, `00_Enstrumantal` |
| 2 | Mel-RoFormer Karaoke | `01_Ana_Vokal`, `02_Arka_Vokal` |
| 3 | Demucs htdemucs_6s | `03_Davul`, `04_Bas`, `05_Gitar`, `06_Piyano`, `07_Diger_Synth_Pad` |
| 4 | MDX23C DrumSep | `08_Davul_Kick` … `13_Davul_Crash` |
| MIDI | Basic Pitch, pYIN, vuruş tespiti | `MIDI/` içinde her stem için `.mid` ve hepsi bir arada `00_Tum_Enstrumanlar.mid` |

MIDI ayrıntıları (kullanıcı sorarsa):
- **Tempo haritası:** Tempo sabit varsayılmaz. AI üretimi (Suno vb.) ve canlı kayıtlarda tempo şarkı boyunca
  kayabiliyor; sabit tempolu MIDI sona doğru vuruşlardan uzaklaşır. Davul vuruşları bulunur ve MIDI'ye vuruş vuruş
  tempo yazılır. DAW'ın grid'i böylece şarkının gerçek vuruşlarına oturur. İlk vuruştan önceki boşluk bir "pickup"
  vuruşu olur. Çoğu DAW, MIDI'yi yeni/boş bir projeye açarken veya içe aktarırken tempo haritasını da alır.
  Klip olarak mevcut bir projeye sürüklendiğinde bazı DAW'lar tempo değişimlerini yok sayar; o zaman proje
  temposunu ortalamaya ayarlamak gerekir ve sona doğru küçük kayma olabilir.
- **Ton:** Şarkının tonu bulunur (ör. E majör), MIDI'ye yazılır. İki perde arasında kalan sesler tona uyan notaya yuvarlanır.
- **Yöntemler:** Her stem için yöntem, gerçek bir şarkıda yan yana ölçülerek seçildi:
  - Bas: pYIN perde takibi. Oktav hatası yapmaz, tekrarlanan notaları vuruşlardan böler.
  - Ana vokal: Basic Pitch, aynı anda tek nota.
  - Arka vokal, gitar, piyano, synth: Basic Pitch, akorlu. Tona uymayan kısa ve zayıf notalar atılır.
- **Davul:** DrumSep parçalarındaki vuruşlardan üretilir: General MIDI kick 36, snare 38, tom 45, hi-hat 42,
  ride 51, crash 49. Kick/snare'den sızan zayıf vuruşlar elenir. `--mod hizli` ve `--mod vokal`'da davul
  parçalanmadığı için davul MIDI'si çıkmaz.
- Her şey 1/16'lık grid'e oturtulur. Swing veya triole dayalı parçalarda grid bunu düzleştirir; kullanıcı ham
  zamanlama isterse `midi_cikar.py "<klasör>" --ham` ile yeniden çıkar.
- MIDI otomatik transkripsiyondur, nota nota kusursuz değildir. En iyi sonuç bas, vokal melodisi ve davulda alınır;
  yoğun akorlu gitar ve pad'lerde fazla/eksik nota olur. Bunu kullanıcıya dürüstçe söyle.

## 1. Kurulum (her seferinde kontrol et, kuruluysa saniyeler sürer)

```
python <skill klasörü>/scripts/kurulum.py
```

Son satırdaki `PYTHON=...` yolunu not al; ayırma bu Python ile çalışır (ortam `~/.stems-venv` içinde).
Ortam yoksa script kurar: 2-4 GB indirir, birkaç dakika sürer. Bunu başlatmadan önce kullanıcıya
kısaca söyle ve komutu arka planda çalıştır. `python` yoksa `py` veya `python3` dene.
ffmpeg uyarısı çıkarsa ve girdi WAV/FLAC değilse, kullanıcıya verilen kurulum komutunu ilet.

## 2. Ayırma

```
<PYTHON> -u <skill klasörü>/scripts/stems.py "<şarkı veya klasör>" [--mod tam|hizli|vokal] [--cikti KLASÖR]
```

- Kullanıcı sadece vokal/altyapı (karaoke, akapella) istiyorsa `--mod vokal` kullan: tek adım, çok daha hızlı.
- Davulun kick/snare'e bölünmesine gerek yoksa `--mod hizli`.
- Aksi halde varsayılan `tam` (bütün enstrümanlar).
- Çıktı, şarkının yanında `<şarkı adı> - Stems` klasörüne gider. Kullanıcı başka yer söylerse `--cikti` ver.
- Birden fazla dosya veya klasör verilebilir, sırayla işlenir.
- MIDI varsayılan olarak çıkar. Kullanıcı istemiyorsa `--midi-yok`.
- Kullanıcı sadece MIDI istiyor ve stem'ler zaten ayrılmışsa ayırmayı tekrarlama:
  `<PYTHON> <skill klasörü>/scripts/midi_cikar.py "<şarkı> - Stems"` yeterli. 1-2 dakika sürer.

Süre: 3-4 dakikalık bir şarkı için `tam` mod, 6 GB'lık bir dizüstü ekran kartında yaklaşık 7-12 dakika,
CPU'da bunun birkaç katı sürer. Komutu **arka planda** çalıştır ve kullanıcıya yaklaşık süreyi söyle.
Model dosyaları ilk kullanımda iner (toplam ~1.5 GB).
Başta görünen "onnxruntime ... CUDA 13" / "Failed to load cublas64_13.dll" uyarıları zararsızdır: bu modellerin
hiçbiri ONNX kullanmaz. Kullanıcıyı bunlarla endişelendirme.

Ekran kartını kullanan başka bir iş (başka bir AI modeli, oyun, video render) aynı anda çalışıyorsa işlem çok
yavaşlar ya da takılmış gibi görünür. 10 dakikada ilk adım bile bitmediyse `nvidia-smi` ile ekran kartını kimin
doldurduğuna bak ve kullanıcıya söyle; kullanıcının diğer işlerini kendi başına durdurma.

## 3. Kontrol (iş bittikten sonra her zaman)

Script her şarkının sonunda bu işleri kendisi yapar:
- RMS seviyesi -50 dB'in altındaki stem'leri boş sayar ve siler. Boş çıkmaları normaldir: o enstrüman şarkıda yoktur.
  Kullanıcı boşları da isterse `--sessizleri-tut` ver.
- Kalan her dosyanın süresini ve örnekleme hızını orijinalle karşılaştırır, clipping'e bakar.
- Vokal + enstrümantal toplamının orijinali ne kadar verdiğini ölçer (-20 dB'den düşük fark = iyi).
- Her MIDI dosyasını açıp okur: nota var mı, şarkı süresini aşıyor mu, bas notaları gerçekten pes mi.
- **Uyum %:** Her MIDI'yi kendi stem'inin sesiyle karşılaştırır (perde içeriği örtüşmesi). %70 üstü iyi,
  %60 altı "dikkat" olarak işaretlenir. Bas için %60-65 normaldir, çünkü pes seslerde ölçüm daha kabadır.
- **Grid uzaklığı:** Davuldaki gerçek vuruşların MIDI grid'ine medyan uzaklığı. 20 ms altı iyi; üstü tempo
  haritasının tutmadığını gösterir ve sorun sayılır.
- Raporu çıktı klasörüne `_kontrol.txt` olarak yazar. Script'in çıkış kodu sorun yoksa 0, varsa 1 olur.

Script bittikten sonra çıktısını (veya `_kontrol.txt`'yi) oku ve kendin de doğrula. Çıkış kodu 1 ise ya da
"SORUN" satırı varsa nedenini incele ve kullanıcıya açıkça söyle; sorunu "tamam" diye geçme.

## 4. Kullanıcıya rapor

Kullanıcı teknik olmayabilir; sade bir dille yaz:
- Çıktı klasörünün yeri (mümkünse dosya yöneticisinde aç).
- Kalan stem'lerin kısa listesi, her biri için bir iki kelime (ör. "Ana vokal, net").
- Silinen boş stem'ler ve nedeni ("şarkıda piyano olmadığı için boş çıktı, silindi").
- MIDI: hangi enstrümanların MIDI'si çıktı, tempo (değişiyorsa aralığı), ton, uyum puanları ve hepsi bir arada
  dosyanın adı. Tempo değişiyorsa DAW'da MIDI'nin tempo haritasını içe almasını söyle.
- Kontrol sonucu: tamam mı, sorun mu?

Ayrımın temiz olup olmadığına kesin kararı ancak dinleyerek verebilirsin; seviyelere bakarak "kusursuz"
deme. Gerekirse kullanıcıdan bir iki stem'i dinlemesini iste.
