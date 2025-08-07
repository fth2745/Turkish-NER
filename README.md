
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

### 📈 Model Gelişimi: Öncesi ve Sonrası Karşılaştırması

#### 📊 BIO Classification Report (B ve I Etiketleri)

**Önce (Epoch 8 / O Etiketi Dahil):**
```
           B       0.61      0.92      0.73    111899
           I       0.47      0.94      0.62     32364
           O       0.99      0.87      0.93    708037
accuracy: 0.88 | macro avg f1: 0.76
```

**Sonra (Sadece B ve I):**
```
           B       0.99      0.92      0.95    111899
           I       0.78      0.97      0.86     32364
accuracy: 0.93 | macro avg f1: 0.91
```

✅ **İyileşme Özeti:**  
- `B` için F1: **%73 → %95**  
- `I` için F1: **%62 → %86**  
- Genel makro ortalama F1: **%76 → %91**

---

#### 📊 Entity Classification Report (O Etiketi Hariç)

**Önce:**
```
macro avg f1: 0.91 | weighted avg f1: 0.92
Öne çıkan sınıf örneği → BİLİM_KÜLTÜR: F1 0.84
```

**Sonra:**
```
macro avg f1: 0.92 | weighted avg f1: 0.92
Öne çıkan sınıf örneği → BİLİM_KÜLTÜR: F1 0.91
```

✅ **İyileşme Özeti:**
- Genel olarak F1 skorlarında artış.
- En zayıf sınıflardan biri olan `BİLİM_KÜLTÜR` sınıfında **%84 → %91** yükseliş.
- Bu durum, hem entity head’in hem embedding yapısının daha sağlam hale geldiğini gösteriyor.

---

#### 📉 Kayıp ve Doğruluk Karşılaştırması

|            | BIO Loss | ENT Loss | BIO Acc | ENT Acc |
|------------|----------|----------|---------|----------|
| **Önce**   | 0.3615   | 0.2879   | 0.7613  | 0.9094   |
| **Sonra**  | 0.1785   | 0.2724   | 0.9300  | 0.9200   |

> BIO task’inde hem doğruluk hem kayıp düzeyinde **belirgin bir iyileşme** gözleniyor.

---

### 📌 Neden "O Etiketi Hariç" Değerlendirme Yaptık?

- `O` etiketi modelin **varlık dışı kelimeleri** ne kadar iyi ayıkladığını gösterir.
- Ancak `O` sınıfı genellikle çok baskındır (örnek: 700k vs 100k). Bu, genel skoru yapay olarak yükseltebilir.
- Bu yüzden sadece `B` ve `I` etiketlerine odaklanmak, modelin gerçekten varlık tespit etme kabiliyetini ölçer.
- Aynı şekilde, entity classification'da da sadece **anlamlı varlık etiketleri** değerlendirilmelidir.
- 
> Bu strateji modelin **varlık sınırlarını daha isabetli belirlemesini**, **false positive'leri azaltmasını** ve **daha hızlı öğrenmesini** sağlar.



