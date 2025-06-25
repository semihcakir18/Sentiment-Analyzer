# Sıfırdan Duygu Analizi
Bu proje, IMDB film yorumları veri setini kullanarak sentiment analizi (duygu analizi) yapan bir derin öğrenme modelinin, hiçbir ana derin öğrenme kütüphanesi (TensorFlow, PyTorch vb.) kullanılmadan, tamamen sıfırdan Python ve NumPy ile inşa edilme sürecini anlatmaktadır.Projedeki Neural.py adlı dosya yine bana ait "Neural-Network-From-Scratch" adlı repodan alınmış ve projenin gereksinimlerine göre üzerine eklemeler yapılmıştır. Projenin temel felsefesi, modern derin öğrenme modellerinin arkasındaki temel mekanikleri manuel olarak kodlayarak derinlemesine anlamaktır.
## Projenin Yolculuğu
Bu proje, basit bir modelle başlayıp karşılaşılan sorunları sistematik olarak analiz ederek ve çözerek daha karmaşık ve güçlü mimarilere evrilmiştir.
### 1. Deneme: Bag-of-Words (Kelimelerin Çantası) ve Basit Ağ
- Yaklaşım: Her bir yorum, içindeki kelimelerin sıklığını temsil eden büyük bir vektöre (örneğin 5000 boyutlu) dönüştürüldü. Kelimelerin sırası ve anlamı tamamen göz ardı edildi. Bu vektörler, sıfırdan yazdığım basit bir feed-forward sinir ağına verildi.
- Sonuç: %84-86 arasında bir doğruluk oranına takılıp kaldı.
- Teşhis: Veriyi PCA ile görselleştirdiğimizde, pozitif ve negatif yorumların vektör uzayında büyük ölçüde iç içe geçtiğini gördüm. Modelin daha fazlasını öğrenmesini engelleyen şey, sinir ağı değil, veri temsilinin yetersizliğiydi.
### 2. Deneme: TF-IDF ve Daha İyi Ön İşleme
- Yaklaşım: Modelin sinyalini güçlendirmek için metin ön işleme adımlarımızı geliştirdim. "Stop words" (anlamsız kelimeler) kaldırıldı, kelimeler köklerine indirgendi (lemmatization) ve basit kelime sayımı yerine TF-IDF (Term Frequency-Inverse Document Frequency) kullanıldı. Bu yöntem, nadir ve önemli kelimelere daha fazla ağırlık verir.
- Sonuç: Doğruluk oranında anlamlı bir artış olmadı.
- Teşhis: Sorunun sadece kelime sıklığıyla ilgili olmadığını anladık. Asıl sorun, modelin kelimelerin anlamını ve bağlamını hiç anlamamasıydı.
### 3. Deneme: Word Embeddings (Kelime Vektörleri) - İlk Atılım
- Yaklaşım: Kelimeleri anlamsal olarak temsil eden, önceden eğitilmiş spaCy kelime vektörlerini kullanmaya karar verdim. Bir yorumu, içindeki tüm kelimelerin vektörlerinin ortalamasını alarak tek bir yoğun vektöre (300 boyutlu) dönüştürdüm.
- Karşılaşılan Zorluk: İlk denememde, spaCy'ye metinleri vermeden önce kendi yazdığım agresif metin temizleme fonksiyonunu kullandım. Bu, spaCy'nin ihtiyaç duyduğu dilbilgisi yapısını bozarak modelin performansını düşürdü.
- Çözüm: Metin temizleme adımını atlayıp, ham metinleri doğrudan spaCy'ye vererek doğru yöntemi buldum. Bu, daha kaliteli ve anlamlı vektörler üretti.
- Sonuç: Modelim artık daha "akıllı" özelliklerle besleniyordu, ancak feed-forward sinir ağı mimarisi, kelimelerin sırasını hala anlayamıyordu. Bu, bu mimarinin ulaşabileceği son noktaydı (~%77 doğruluk).
### 4. Deneme: Sıfırdan LSTM (Nihai Model) - Zirve
- Yaklaşım: Kelime sırasını anlamanın tek yolu, Recurrent Neural Network (RNN) kullanmaktı. Projenin ruhuna sadık kalarak, en popüler ve güçlü RNN türü olan LSTM (Long Short-Term Memory) katmanını tamamen sıfırdan kodlamaya karar verdim.
- Sıfırdan İnşa Edilenler:
     - EmbeddingLayer: Kelime indekslerini öğrenilebilir vektörlere dönüştüren katman.
     - LSTMLayer: Dört kapı (forget, input, output, gate) ve iç hafıza (cell state) yapısıyla birlikte, zaman içinde geri yayılımı (Backpropagation Through Time) da içeren tam bir LSTM katmanı.
     - SimpleTokenizer: Metinleri sayısal dizilere dönüştüren basit bir tokenizer.
     - Özelleştirilmiş Eğitim Döngüsü: Bu yeni sıralı mimariyi eğitebilmek için, Adam optimizasyon algoritmasını ve Gradient Clipping (patlayan gradyanları önleme) tekniğini manuel olarak uygulayan özel bir eğitim döngüsü.
- Sonuç: Bu karmaşık ama güçlü mimari, kelimelerin sırasını ve bağlamını anlayarak projenin nihai hedefine ulaştı.

## 1.Proje Yapısı
Sentiment-Analyzer/     
├── data/                     
├── models/                   
├── src/  
│   ├── __ init__.py           
│   ├── data_loader.py      
│   ├── download_nltk_data.py   
│   ├── Neural.py          
│   ├── preprocessing.py   
│   ├── tokenizer.py        
│   ├── train_lstm.py       
│   ├── vectorizer_spacy.py   
│   │    
│   ├── not_being_used_anymore/    
│   │   ├── train_model.py    
│   │   └── ...    
│   │    
│   └── tests_and_inspections/    
│       ├── __ init__.py        
│       ├── inspect_data.py   
│       └── test_setup.py        
│    
├── .gitignore                    
├── main.py                    
├── README.md                 
└── requirements.txt  

# 2.Kurulum ve Çalıştırma
Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.
## 2.1. Projeyi Klonlama
```bash
git clone https://github.com/semihcakir18/Sentiment-Analyzer
cd Sentiment-Analyzer
```
    
## 2.2. Bağımlılıkları Kurma
Projenin izole bir ortamda çalışması için bir sanal ortam oluşturmanız şiddetle tavsiye edilir.
``` bash
python -m venv venv
```
### Sanal ortamı aktif et 
 - Windows için
```bash
venv\Scripts\activate
```
-  MacOS/Linux için:
```bash
source venv/bin/activate
```
## 2.3 Gerekli kütüphaneleri requirements.txt dosyasından kur
```bash
pip install -r requirements.txt
```
## 2.4 Kurulum Komutunu Çalıştırma (Tek Seferlik)
Bu komut, projenin ihtiyaç duyduğu spaCy ve NLTK modellerini otomatik olarak indirecektir.
```bash
python main.py setup
```
## 2.5 Gerekli veriyi indirme
Bu linkteki klasörü ayıklayıp __data__ klasörüne atmanız gereklidir
https://ai.stanford.edu/~amaas/data/sentiment/

# 3. Kurulumu ve Veriyi İnceleme (Opsiyonel)
Bu zorunlu bir adım değil ama kurulumdan emin olabilmek için önemli bir adımdır. Aşağıdaki komutu çalıştırdığınızda önce pipeline'ın düzgün çalışıp çalışmadığı kontrol edilir , sonrasında ise veriler üzerinden vektör grafikleri oluşturulur.

```bash
python main.py inspect
```
# 4. Modeli Eğitme
Kurulum tamamlandıktan sonra, aşağıdaki komutla sıfırdan yazdığım LSTM modelinin eğitimini başlatabilirsiniz.

Not: Bu işlem, özellikle ilk çalıştırmada vektör önbelleği oluşturulurken ve Python'da sıfırdan hesaplama yapıldığı için oldukça yavaş olacaktır. Bu beklenen bir durumdur.

```bash
python main.py train
```

### Nihai Performans
Sıfırdan inşa ettiğim LSTM modeli, kelime sırasını ve bağlamı anlayarak ~%85'lik bir doğruluk oranına ulaşmayı başarmıştır. Bu, basit istatistiksel modellerin sınırlarını aşan ve projenin temel hedefine ulaştığını gösteren bir başarıdır.


Bu proje, bir derin öğrenme modelinin sadece nasıl kullanılacağını değil, aynı zamanda en temel seviyede nasıl çalıştığını anlamak isteyenler için örnek açısından yapılmıştır.