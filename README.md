
# 🇹🇷 Multi-Task Turkish NER (BERT + BiLSTM)

Bu proje, Türkçe metinler için gelişmiş bir Varlık Tanıma (NER) sistemi içerir. Model, BERT'in derin anlamsal anlama yeteneğini BiLSTM'in sıralı desen yakalama gücüyle birleştirir. En temel özelliği, standart B-I-O etiketlerini tahmin ederken aynı anda bu varlıkların üst düzey kategorilerini sınıflandıran çok görevli (multi-task) bir mimariye sahip olmasıdır.

---

## 🚀 Temel Özellikler ve Konseptler

### ✅ Çok Görevli Öğrenme (Multi-Task Learning)

Model, tek bir girdi üzerinden iki görevi eş zamanlı olarak öğrenir:

- **BIO Tespiti:** Kelimenin bir varlık başlangıcı (B), içi (I) ya da dışı (O) olup olmadığını belirler.
- **Varlık Sınıflandırması:** Tespit edilen varlığın KİŞİ, YER, KURUM vb. gibi üst düzey kategorisini tahmin eder.

### 🧱 Hibrit Model Mimarisi

- **BERT**: [`dbmdz/bert-base-turkish-128k-cased`](https://huggingface.co/dbmdz/bert-base-turkish-128k-cased) modeli ile bağlamsal kelime temsilleri üretilir.
- **BiLSTM**: BERT'ten gelen çıktılarla ileri ve geri yönde sıralı bilgi işlenir.

### ❄️ Dinamik Katman Dondurma

Eğer bir görev (örneğin BIO) için doğrulama kaybı artarsa, o göreve ait sınıflandırıcı katman geçici olarak dondurulur. Bu, görevler arası etkileşimi dengeleyerek modelin kararlılığını artırır.

### 🧲 Triplet Loss ile Gömme (Embedding) Öğrenimi

Aynı kategoriye ait varlıkların gömme uzayında birbirine yakın, farklı kategorilerin ise uzak olması sağlanır. Bu amaçla `TripletMarginLoss` kullanılır.

### ⚙️ Verimli Eğitim Teknikleri

- **AMP (Mixed Precision Training):** Daha hızlı ve bellek verimli eğitim için `torch.cuda.amp`.
- **BucketBatchSampler:** Benzer uzunluktaki cümleler aynı batch'e alınarak padding azaltılır.
- **Sınıf Dengesizliği:** `CrossEntropyLoss` içinde sınıf ağırlıkları kullanılarak çözülür.

---

## 🧠 Model Mimarisi

```
          Girdi Cümlesi (Tokenlar)
                      ↓
     Tokenization (BERT Tokenizer)
                      ↓
BERT (dbmdz/bert-base-turkish-128k-cased)
                      ↓
        Layer Normalization
                      ↓
         BiLSTM (2 Katmanlı)
                      ↓
┌─────────────────────┴─────────────────────┐
│                                           │
↓                                           ↓
BIO Sınıflandırma Head             Varlık Sınıflandırma Head
(Dense -> ReLU -> Dropout ->       Dense -> ReLU -> Dropout ->
 Linear)                           Linear
│                                           │
↓                                           ↓
BIO Logitleri (B, I, O)             Varlık Logitleri (KİŞİ, YER, ...)
│                                           │
↓                                           ↓
BIO Kaybı (CrossEntropyLoss)       Varlık Kaybı + Triplet Loss
```

---

## 📂 Veri Seti ve Ön İşleme

Model `.jsonl` formatında veri bekler. Her satır bir cümleyi temsil eder.

### ✅ Beklenen Format

```json
{
  "tokens": ["Mustafa", "Kemal", "Atatürk", "Anıtkabir'de", "yatıyor", "."],
  "ner_tags": ["B-person", "I-person", "I-person", "B-location", "O", "O"]
}
```

### 🔧 Ön İşleme Adımları

1. **Etiket Gruplama:** Örneğin `B-government_person` gibi detaylı etiketler `KİŞİ` gibi üst düzey kategorilere indirgenir.
2. **Etiket Ayırma:** Etiketler `bio_tags` ve `entity_tags` olmak üzere ikiye ayrılır.
3. **Tokenizasyon:** BERT tokenizer ile alt kelimelere ayrılır ve etiketler hizalanır.
4. **Maskeleme:** `O` etiketli ve düşük güvenli tahminler (`bio_threshold` altında) `-100` ile maskelenir.

---

## ⚙️ Kurulum ve Kullanım

### 1. Projeyi Klonlayın

```bash
git clone https://github.com/kullanici/proje-adi.git
cd proje-adi
```

### 2. Gerekli Kütüphaneleri Yükleyin

```bash
pip install torch pandas numpy transformers scikit-learn torch-optimizer
```

> Not: `requirements.txt` dosyası kullanmanız önerilir.

### 3. Veri Setini Hazırlayın

Veri setinizi yukarıda açıklanan `.jsonl` formatında hazırlayın. `main()` fonksiyonu içinde dosya yolunu aşağıdaki şekilde düzenleyin:

```python
full_data = load_dataset_from_json(
    "path/to/your/dataset.jsonl",
    config["tokens_col"],
    config["tags_col"],
    entity_group_map=entity_group_map
)
```

### 4. Yapılandırmayı Düzenleyin

```python
config = {
    "model_name": "dbmdz/bert-base-turkish-128k-cased",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 3e-5,
    "num_epochs": 10,
    ...
}
```

### 5. Eğitimi Başlatın

```bash
python ner_trainer.py
```

---

## 📊 Değerlendirme

- BIO ve Entity görevleri ayrı ayrı değerlendirilir.
- Aşağıdaki metrikler her görev için ayrı hesaplanır:
  - Accuracy
  - Precision / Recall / F1 (macro average)
- `classification_report` ile B/I ve her sınıf için detaylı sonuç sunulur.
- En iyi model test setinde yeniden değerlendirilir.

---

## 🧩 Yapılandırma Parametreleri

| Parametre | Açıklama |
|-----------|----------|
| `model_name` | Kullanılacak BERT modeli adı. |
| `max_length` | Tokenizer'ın maksimum uzunluğu. |
| `batch_size` | Eğitimde kullanılacak batch boyutu. |
| `learning_rate` | Öğrenme oranı. |
| `num_epochs` | Toplam epoch sayısı. |
| `early_stopping_patience` | Doğrulama kaybı iyileşmezse eğitimin durdurulacağı epoch sayısı. |
| `triplet_loss_weight` | TripletLoss'un toplam kayba katkı oranı. |
| `bio_freeze_patience` | BIO kaybı bu kadar epoch artarsa BIO katmanı dondurulur. |
| `bio_threshold` | B veya I tahmininin güven skoru bu eşiğin altındaysa dikkate alınmaz. |

---

## ❓ Neden `O` Etiketi İçin Özel İşlem Yapıyoruz?

Çoğu NER veri kümesinde "O" (Outside) etiketi, B ve I etiketlerine göre çok daha fazladır. Bu durum sınıf dengesizliğine yol açar. Bizim yaklaşımımız:

- `O` etiketli tokenların bir kısmını kayıp fonksiyonundan dışlamak,
- Bu dışlamayı tahminin düşük güvenli (örneğin softmax skoru düşük) olması durumunda yapmak,
- Böylece modelin sadece emin olduğu B/I tahminlerine odaklanmasını sağlamak.

> Bu strateji modelin **varlık sınırlarını daha isabetli belirlemesini**, **false positive'leri azaltmasını** ve **daha hızlı öğrenmesini** sağlar.


