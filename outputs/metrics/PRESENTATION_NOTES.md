# Dönem Ortası Sunumu — Hangi Dosyalar, Nasıl Yorumlanır?

## 1. outputs/metrics altındaki dosyalar sunum için yeterli mi?

**Evet.** Utility (accuracy) kısmını anlatmak için bu klasör yeterli. İstersen yanına sadece **non-IID’i göstermek** için `outputs/splits/split_summary_alpha0.1.txt` (veya ilk birkaç satır) da ekleyebilirsin.

| Dosya | Sunumda kullanım |
|-------|-------------------|
| `utility_summary_latest.txt` | Tek slide: ortalama accuracy, std, min/max, client sayısı. |
| `utility_summary_alpha0.1_seed42.txt` | Detay slide: client bazlı accuracy tablosu. |
| `client_accuracy.csv` | Ham veri; “sonuçlar bu CSV’den üretildi” demek için. |
| `utility_summary.json` | Grafik çizeceksen (örn. bar chart) programatik kullanım. |

---

## 2. Senin sonuçların (alpha=0.1, seed=42) — sayılar

- **15 client**, Dirichlet α=0.1, **1 epoch** local training.
- **Ortalama accuracy:** %16,2 (0,162).
- **Std:** ~%5,8.
- **Min:** %10 (Client 0, 4) — **Max:** %29,7 (Client 10).
- Client’lar **5–9 sınıf** prototype gönderiyor (non-IID: bazı sınıflar yok).

---

## 3. Nasıl yorumlanır? (Sunumda söyleyebileceğin cümleler)

### Accuracy neden düşük?

- **Sadece 1 local epoch** kullandık; bu bir **baseline / hızlı deney** ayarı.
- Literatürde genelde daha fazla epoch ile %50–70’lere çıkılır; biz önce pipeline’ı (one-shot FL + prototype) kurup utility’yi ölçtük.
- Sunumda: *“Baseline utility’yi 1 epoch ile raporluyoruz; ortalama local test accuracy %16,2. İleride epoch sayısını artırarak utility’yi yükseltip privacy–utility trade-off’unu inceleyeceğiz.”*

### Client’lar arası fark (heterogeneity)

- **Min %10 – max %29,7:** Client’lar arasında belirgin fark var.
- **Neden?** Veri Dirichlet(α=0.1) ile bölündüğü için:
  - Bazı client’lar az sınıfa sahip (örn. 5 sınıf), bazıları daha fazla (9 sınıf).
  - Sample sayıları da farklı (654 – 7822).
- Sunumda: *“Client-level accuracy dağılımı heterojen (min %10, max %29,7); bu, non-IID veri bölümünü (Dirichlet α=0,1) yansıtıyor ve ileride class-presence inference gibi saldırıları anlamlı kılıyor.”*

### Non-IID’i nasıl gösterirsin?

- `outputs/splits/split_summary_alpha0.1.txt` içinde “classes_present= 5” … “9” ve “n= 654” … “7822” var.
- Sunumda kısa bir tablo veya 1–2 cümle: *“Her client’ta 5–9 sınıf bulunuyor; bazı sınıflar bazı client’larda yok. Bu da sunucunun sadece gelen prototype’lardan ‘hangi sınıflar vardı?’ çıkarımı yapmasını anlamlı kılıyor.”*

---

## 4. Özet tablo (slayta koyabileceğin)

| Metrik | Değer |
|--------|--------|
| Ortalama accuracy (15 client) | **%16,2** |
| Std | %5,8 |
| Min / Max accuracy | %10,0 / %29,7 |
| Local epoch | 1 (baseline) |
| Non-IID | Dirichlet α=0,1 |
| Sınıf sayısı (client başına) | 5–9 (prototype gönderilen) |

---

## 5. İsteğe bağlı: tek slaytta “ne yaptık, ne ölçtük”

- **Ne yaptık:** One-shot personalized FL; her client aynı global modelle başladı, sadece kendi verisiyle 1 epoch eğitti, **sadece class-wise prototype’ları** (512-dim) paylaştı; aggregation yok.
- **Ne ölçtük:** Her client’ın kendi local test set’indeki accuracy’sini; ortalaması **%16,2**, client’lar arası dağılım geniş (heterojen).
- **Neden bu tasarım:** Prototype paylaşımı üzerinden sonraki aşamada **membership** ve **class presence** inference saldırılarını inceleyeceğiz; non-IID ve düşük epoch sayısı baseline’ı oluşturuyor.

Bu metni, yukarıdaki tablo ve `utility_summary_latest.txt` / `utility_summary_alpha0.1_seed42.txt` ile birlikte kullanırsan **outputs/metrics** (ve istersen split özeti) dönem ortası sunumu için yeterli olur.
