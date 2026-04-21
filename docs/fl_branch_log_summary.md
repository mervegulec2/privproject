# Logs/Artifacts snapshot (from `fl` branch)

Bu doküman, daha önce `fl` branch’inde üretilmiş **baseline utility/log** çıktılarını,
bu branch’e “bilgi olarak” taşımak için özetler.  
Amaç: saldırı branch’inde repo’yu şişirmeden (npz/png gibi binary artifact’leri eklemeden) sunum/rapor metinlerini korumak.

## Kaynak

`fl` branch’inde şu dosyalar takiplenmişti (bazıları gitignore kapsamına girdiği için bu branch’e doğrudan mergelemek uygun değil):

- `outputs/metrics/PRESENTATION_NOTES.md`
- `outputs/metrics/client_accuracy.csv`
- `outputs/metrics/utility_summary_latest.txt`
- `outputs/metrics/utility_summary_alpha0.1_seed42.txt`
- `outputs/splits/split_summary_alpha0.1.txt`
- (binary örnekler) `outputs/client_protos_*/client_*_prototypes.npz`, `outputs/class_distribution.png`

## Sunum için özet (alpha=0.1, seed=42)

`outputs/metrics/PRESENTATION_NOTES.md` içeriğinden öne çıkanlar:

- 15 client, Dirichlet \( \alpha = 0.1 \), **1 epoch** local training (baseline / hızlı deney).
- Ortalama local test accuracy: **%16,2** (0.162)
- Std: ~%5,8
- Min: %10 — Max: %29,7
- Client başına gönderilen prototype sınıf sayısı: **5–9** (non-IID nedeniyle bazı sınıflar yok)

## Not

Bu branch’in threat model’i gereği saldırı kodu **yalnızca server-visible artifact**’lerle çalışır.  
Baseline utility log’ları bu dokümana “rapor” olarak taşındı; ham binary artifact’ler (npz/png) bu branch’e eklenmedi.

