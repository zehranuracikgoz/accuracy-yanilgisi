# Makine Öğrenmesinde Accuracy Yanılgısı

Dengesiz veri setlerinde yüksek accuracy'nin yanıltıcı olabileceğini gerçek bir veri seti üzerinde gösteren bir analiz projesi. Accuracy yerine hangi metriklerin kullanılması gerektiği incelenmektedir.


## Veri Seti

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284.807 kredi kartı işlemi
- 492 dolandırıcılık vakası (%0.17)
- Dengesizlik oranı: 1'e 577


## Proje İçeriği

1. Veri yükleme ve sınıf dağılımı analizi
2. Logistic Regression modeli eğitimi
3. Accuracy, Precision, Recall, F1 Score ve ROC-AUC hesaplama
4. Classification report ve confusion matrix analizi
5. Metrik karşılaştırma görsellerinin üretilmesi


## Sonuçlar

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.9992 ⚠️ |
| Precision | 0.8289 |
| Recall | 0.6429 |
| F1 Score | 0.7241 |
| ROC-AUC | 0.9560 |

Accuracy − F1 farkı **0.2750** — 98 dolandırıcılık işleminden 35'i tespit edilemedi.


## Çalıştırma

Veri seti Kaggle'dan indirilerek proje klasörüne eklenmelidir.

```bash
pip install -r requirements.txt
python accuracy_yanilgisi.py
```


## Kullanılan Teknolojiler

`Python` · `Pandas` · `Scikit-learn` · `Matplotlib` · `Seaborn`

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@zehranuracikgoz/makine-%C3%B6%C4%9Frenmesinde-accuracy-yan%C4%B1lg%C4%B1s%C4%B1-78f86b5c2c4a)
