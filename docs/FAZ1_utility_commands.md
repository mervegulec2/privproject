# FAZ 1 — Utility (Accuracy) Nasıl Çalıştırılır?

Bu rehber: önce baseline FL’i çalıştırıp `client_accuracy.csv` üretmek, sonra utility özetini hesaplayıp **dosyalara** kaydetmek.

---

## 1. Ortam

Proje kökünde sanal ortamı kullan:

```bash
cd /Users/mervegulec/Desktop/privproject
source .venv/bin/activate
```

*(İstersen `source` yerine her komutta `.venv/bin/python` kullanabilirsin.)*

---

## 2. Split’lerin Hazır Olması

Alpha=0.1, seed=42 için split dosyası yoksa üret:

```bash
.venv/bin/python -m src.data.check_split --alpha 0.1 --seeds 42
```

**Çıktı:**  
`outputs/splits/cifar10_dirichlet_a0.1_seed42.npy`  
*(Zaten varsa bu adımı atlayabilirsin.)*

---

## 3. Baseline FL’i Çalıştır (Client Accuracy Üretir)

**Tek komutla (önerilen):**

```bash
bash scripts/run_baseline.sh
```

Bu script:

1. Gerekirse split’i üretir.
2. Server’ı arka planda başlatır.
3. 15 client’ı sırayla başlatır (hepsi bitene kadar bekler).
4. Bittiğinde server da kapanır.

**Üretilen dosyalar:**

| Dosya | Açıklama |
|-------|----------|
| `outputs/metrics/client_accuracy.csv` | Her client için: alpha, seed, client_id, local_test_accuracy, num_train_samples, num_proto_classes |
| `outputs/client_protos_a0.1_s42/client_0_prototypes.npz` … `client_14_prototypes.npz` | Her client’ın class-wise prototype’ları (FAZ 2 için) |

**Önemli:**  
`client_accuracy.csv` **append** modunda yazılır. Aynı alpha/seed ile tekrar çalıştırırsan aynı run için satırlar **tekrar eklenir**. Temiz bir run istiyorsan önce bu dosyayı sil veya yedekle:

```bash
rm -f outputs/metrics/client_accuracy.csv
bash scripts/run_baseline.sh
```

---

## 4. Utility Özetini Hesapla ve Dosyaya Kaydet

CSV üretildikten sonra utility (accuracy) özetini hesaplatıp **hep aynı klasöre** kaydettir:

```bash
.venv/bin/python -m src.eval.utility_summary
```

**Varsayılan davranış:**

- Okur: `outputs/metrics/client_accuracy.csv`
- Yazar: `outputs/metrics/` altına aşağıdaki dosyalar.

**Üretilen dosyalar:**

| Dosya | İçerik |
|-------|--------|
| `outputs/metrics/utility_summary_alpha0.1_seed42.txt` | Bu run için: mean, std, min, max accuracy + client bazlı liste |
| `outputs/metrics/utility_summary_latest.txt` | En güncel run’ın kısa özeti (veya CSV’de birden fazla run varsa hepsinin özeti) |
| `outputs/metrics/utility_summary.json` | Tüm özetler (programatik kullanım için): runs, overall, path’ler |

**Sadece belirli alpha/seed için özet istersen:**

```bash
.venv/bin/python -m src.eval.utility_summary --alpha 0.1 --seed 42
```

**Farklı bir CSV veya çıktı klasörü:**

```bash
.venv/bin/python -m src.eval.utility_summary --csv outputs/metrics/client_accuracy.csv --output-dir outputs/metrics
```

Terminalde de kısa bir özet basılır; ayrıntılar **her zaman** yukarıdaki dosyalarda kalır.

---

## 5. Özet: Sırayla Yapıştıracağın Komutlar

Tek seferde temiz bir utility run’ı için:

```bash
cd /Users/mervegulec/Desktop/privproject
source .venv/bin/activate

# (İsteğe bağlı) Eski CSV’yi temizle ki tek run olsun
rm -f outputs/metrics/client_accuracy.csv

# Baseline: server + 15 client (alpha=0.1, seed=42)
bash scripts/run_baseline.sh

# Utility özetini hesapla ve outputs/metrics/ altına kaydet
.venv/bin/python -m src.eval.utility_summary
```

Bundan sonra:

- **Client-level ve ortalama accuracy** → `outputs/metrics/utility_summary_alpha0.1_seed42.txt` ve `utility_summary_latest.txt`
- **Ham veri** → `outputs/metrics/client_accuracy.csv`
- **Programatik kullanım** → `outputs/metrics/utility_summary.json`

---

## 6. Çıktıların Nerede Durduğu (Özet)

| Ne | Konum |
|----|--------|
| Split (alpha=0.1, seed 42/123/999) | `outputs/splits/cifar10_dirichlet_a0.1_seed*.npy` |
| Client accuracy (ham) | `outputs/metrics/client_accuracy.csv` |
| Utility özet (metin) | `outputs/metrics/utility_summary_alpha0.1_seed42.txt`, `utility_summary_latest.txt` |
| Utility özet (JSON) | `outputs/metrics/utility_summary.json` |
| Prototipler (FAZ 2 için) | `outputs/client_protos_a0.1_s42/*.npz` |

Tüm bu çıktılar proje içinde **dosya olarak** saklanır; FAZ 2’ye geçmeden önce utility’yi bu dosyalardan inceleyebilirsin.
