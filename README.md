# Prototype-sharing FL — Privacy Attacks (strict server-visible)

Bu repo/branch, **class-wise prototype sharing** (FedProto benzeri artifact) kullanan FL protokolleri için,
**yalnızca server-visible artifact**’lerle çalışan privacy attack baseline’larını içerir.

## Threat model (strict server)

- **Passive honest-but-curious server**
- Saldırı kodu sadece server’a gerçekten gönderilen alanları kullanabilir.
- Bu kural teknik olarak `meta.json["sent_fields"]` üzerinden enforce edilir (bkz. `src/attacks/common_io.py`).

## Neler var?

- **CPA (Class Presence Attack)**:
  - **Trivial / protocol leakage baseline**: “prototype anahtarı gönderildiyse sınıf vardır”.
  - Learned CPA için feature hook’ları (geleceğe yönelik).
- **MIA (Membership Inference Attack)**:
  - **Feasibility-first evaluator**: mevcut strict protokolde record-level MIA’nın neden sınırlı/unsupported olduğunu raporlar.
  - Çıktılar daha somut: `status`, `reason`, `required_future_artifacts`.
- **Audit script**:
  - `tools/audit_attacks.py` client upload’larında local-only/forbidden alan olup olmadığını tarar.

Detaylı teknik not: `src/attacks/attack.md`

## One-shot ve personalized etkisi

- Bu dalın odaklandığı yüzey: **one-shot upload snapshot** (server’ın gördüğü tek seferlik artifact).
- Protokol personalized ise leakage yüzeyi ikiye ayrılabilir:
  - **pre-upload surface**: server-visible artifact (bu dal)
  - **post-personalization surface**: client’ın yerelde yaptığı adımlar (strict server modelde server’a görünmez)

## Nasıl çalıştırılır?

Saldırı runner’ları:

```bash
source .venv/bin/activate

# CPA
python run_cpa.py runs/<run_id> --save

# MIA feasibility
python run_mia.py runs/<run_id>

# Audit: forbidden keys kontrolü
python tools/audit_attacks.py runs/<run_id>
```

## Literatür (önerilen sıra)

1. **FedProto** — current communication artifact surface (class-wise prototype sharing)  
2. **Lixu Wang et al., “Eavesdrop the Composition Proportion of Training Labels in Federated Learning”, 2019** — protocol leakage / CPA  
3. **Nasr et al., “Comprehensive Privacy Analysis of Deep Learning: Standalone and Federated Learning”, 2019** — FL’de MIA çerçevesi  
4. (Genel arka plan) Shokri et al., “Membership Inference Attacks against Machine Learning Models”, 2017

## MIA “future extension” notu

Mean/var tabanlı Mahalanobis-benzeri skorlar gibi yaklaşımlar ancak şu tip ek server-visible artifact’ler varsa anlamlıdır:

- per-class mean/variance (veya mean/cov)
- veya server-visible bir query representation \(z\) / per-sample summary

Mevcut baseline bunu “future extension” olarak ele alır; strict protokol altında önce feasibility raporlanır.
