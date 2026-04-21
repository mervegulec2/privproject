# Privacy Attacks (feature/privacy-attacks)

Bu dosya, `feature/privacy-attacks` dalı için attack altyapısının amacı, yapılanlar,
teknik detaylar ve ilgili literatür bağlantılarını Türkçe olarak özetler.

## Amaç

Proje hedefi: sadece server-visible artifact'lerle çalışacak şekilde saldırı altyapısını
kurgulamak ve iki temel baseline sunmak:

- CPA (Class Presence Attack) — protocol leakage / trivial baseline
- MIA (Membership Inference Attack) — feasibility analizi ve future-hook'lar

Threat model: passive honest-but-curious server — saldırı sadece server'a gerçekten gönderilen
verilerle yapılabilir. Server'a gitmeyen hiçbir local-only bilgi kullanılmayacaktır.

## Yapılanlar (kısa)

- `src/attacks/common_io.py`: `runs/{run_id}/meta.json` ve per-round client upload / server artifact
  dosyalarını okuyan yardımcılar. Sadece `meta.json["sent_fields"]` tarafından izin verilen
  alanları döndürür.

- `src/attacks/common_metrics.py`: ROC-AUC, PR-AUC, TPR@FPR hesapları ve aggregation helper'ları;
  degenerate (tek-sınıf/boş) durumlar için koruma eklendi.

- `src/attacks/common_types.py`: `AttackResult`, `FeasibilityReport`, `CPADatasetRow`, `MIADatasetRow` gibi
  dataclass/NamedTuple tanımları.

- `src/attacks/cpa/`:
  - `cpa_dataset.py`: `(client, class)` bazlı dataset builder (`build_cpa_dataset`).
  - `cpa_trivial.py`: trivial CPA (prediction = 1[k in sent_keys]).
  - `cpa_eval.py`: CPA sonuçlarını toplayıp raporlayan evaluator.
  - `cpa_features.py`: future learned-CPA için feature hook'ları.

- `src/attacks/mia/`:
  - `mia_dataset.py`: server-visible sinyallere göre feasibility analizi.
  - `mia_baselines.py`: scorer interface ve feasibility helper'lar.
  - `mia_features.py`: feature mode tanımları (current vs future).
  - `mia_eval.py`: feasibility + placeholder evaluator.

- Runner ve yardımcılar:
  - `run_cpa.py`, `run_mia.py`, `run_attacks.py`
  - `configs/*.yaml`
  - `tools/audit_attacks.py`: yüklemelerde local-only alan kullanımı olup olmadığını tarayan script.

- `attack_outputs/` dizin yapısı eklendi (`cpa/`, `mia/`).

## Teknik detaylar ve formüller

Aşağıda kullanılan temel matematiksel ifadeler ve nasıl uygulandıkları özetlenmiştir.

**CPA (Trivial protocol-leakage)**

Trivial CPA tahmini, her client `c` için sınıf `k`'nın varlığını şu şekilde tahmin eder:

$$
\hat y_{c,k}^{\mathrm{trivial}} = \mathbf{1}[k \in \mathrm{keys}(\mathrm{prototype\_dict}_c)],
$$

Burada `prototype_dict_c` client'ın server'a gönderdiği sınıf-prototip sözlüğüdür. `1` gönderilmişse
sınıfın client'ta bulunması beklenir.

Evaluasyon için kullanılan skor (örnek): ROC-AUC ve PR-AUC. Ancak dataset tek sınıf içeriyorsa
bu metrikler hesaplanamaz — bu nedenle `common_metrics` içinde degenerate-case koruması vardır.

**MIA (Feasibility, strict server-visible protocol)**

Mevcut protokol yalnızca sınıf-başına prototipler ve sayımlar (counts) gönderiyorsa,
membership inference için şu sınırlamalar vardır:

- Server, örnek-seviyesinde embedding/özellik görmediği için doğrudan örnek-temelli MIA zor.
- Ancak sınıf-başına dağılım/ortalama/var gibi istatistikler ileride gönderilirse MIA daha mümkün hale gelir.

Gelecekte eklenecek scoring (şimdilik TODO): mean/var tabanlı Mahalanobis-like skorlar,
örnek bir skor formu (diagonal covariance varsayımıyla):

$$
s_{diag}(z, c, k) = -\sum_d \frac{(z_d - \mu_{c,k,d})^2}{\sigma^2_{c,k,d} + \epsilon},
$$

burada $z$ target örnek temsilidir, $\mu_{c,k}$ ve $\sigma^2_{c,k}$ client'ın sınıf-$k$'si için
server-visible ortalama ve varyanstır.

## Konfigürasyon

- `configs/attacks.yaml`: genel ayarlar (run glob, output dir, report format)
- `configs/cpa.yaml`: `num_classes`, `evaluate_per_class`, `protocol_leakage_mode`
- `configs/mia.yaml`: `strict_server_mode`, `future_extended_mode`, `allowed_features`

## Çıktı formatı

Her run için attack çıktıları `run_dir/attacks/` altına JSON ve CSV olarak kaydedilir. Ayrıca
proje seviyesinde `attack_outputs/cpa/` ve `attack_outputs/mia/` dizinleri yer alır.

## Literatür (seçme)

- Shokri, R., Stronati, M., Song, C., Shmatikov, V., "Membership Inference Attacks against Machine Learning Models", 2017.
- Nasr, M., Shokri, R., Houmansadr, A., "Comprehensive Privacy Analysis of Deep Learning: Standalone and Federated Learning", 2019.
- Wang et al., "Protocol leakage and privacy risks in prototype-based sharing", 2019. (prototypical model leakage discussion)

(Not: yukarıdaki başlıklar literatürde bilinen çalışmalara karşılık gelmektedir; tam bibliyografik referanslar README/docs içinde genişletilebilir.)

## Nasıl kullanılır (hızlı)

```bash
# CPA çalıştırma (örnek)
source .venv/bin/activate
python run_cpa.py runs/cifar10_a0.1_s42_c2_r1 --save

# MIA feasibility
python run_mia.py runs/cifar10_a0.1_s42_c2_r1

# Audit (forbidden local-only fields)
python tools/audit_attacks.py runs/cifar10_a0.1_s42_c2_r1
```

## Son notlar ve TODO'lar

- `cpa_dataset.py` ve `cpa_features.py` tamamlandı; learned CPA için ek feature'lar ileride eklenecek.
- `mia` tarafı şu an feasibility-first: gerçek MIA pipeline'ı için sample-level representation'ın
  server-visible hale gelmesi gerekiyor (veya server'ın publish ettiği summary statistics artırılmalı).
- Kritik kural: hiçbir attack modülü local-only veriyi kullanmamalıdır. `tools/audit_attacks.py`
  bu kuralı taramak için kullanılabilir.

---

Bu dosyayı güncel tutacağım; istersen adım adım değişiklikleri commitleyip branch'e push edebilirim.
