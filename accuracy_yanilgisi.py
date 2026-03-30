import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler


plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "normal": "#4A90D9",
    "fraud": "#E05C5C",
    "accent": "#5C6BC0",
    "highlight": "#FF7043",
    "gray": "#B0BEC5",
}


def veriyi_yukle(dosya_yolu: str) -> pd.DataFrame:
    print("=" * 60)
    print("1. VERİ YÜKLENİYOR")
    print("=" * 60)

    df = pd.read_csv(dosya_yolu)
    print(f"Toplam kayıt sayısı : {len(df):,}")
    print(f"Sütun sayısı        : {df.shape[1]}\n")

    dagilim = df["Class"].value_counts()
    toplam  = len(df)

    print("Sınıf Dağılımı:")
    print(f"  Normal işlem (0)       : {dagilim[0]:>7,}  (%{dagilim[0]/toplam*100:.2f})")
    print(f"  Dolandırıcılık (1)     : {dagilim[1]:>7,}  (%{dagilim[1]/toplam*100:.4f})")
    print(f"  Dengesizlik oranı      : 1 : {dagilim[0]//dagilim[1]}\n")

    return df


def sinif_dagilimini_gorsellestir(df: pd.DataFrame, cikti: str = "class_distribution.png"):
    print("2. SINIF DAĞILIMI")

    dagilim = df["Class"].value_counts().sort_index()
    etiketler = ["Normal (0)", "Dolandırıcılık (1)"]
    renkler   = [COLORS["normal"], COLORS["fraud"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Kredi Kartı Veri Seti — Sınıf Dağılımı", fontsize=15, fontweight="bold", y=1.02)

    bars = axes[0].bar(etiketler, dagilim.values, color=renkler, width=0.5, edgecolor="white")
    axes[0].set_title("İşlem Sayısı", fontsize=12)
    axes[0].set_ylabel("Kayıt Sayısı")
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, dagilim.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    yuzde = dagilim.values / len(df) * 100
    bars2 = axes[1].bar(etiketler, yuzde, color=renkler, width=0.5, edgecolor="white")
    axes[1].set_title("Yüzde Dağılımı (log ölçeği)", fontsize=12)
    axes[1].set_ylabel("Yüzde (%)")
    axes[1].set_yscale("log")
    for bar, val in zip(bars2, yuzde):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val * 1.3,
                     f"%{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(cikti, bbox_inches="tight")
    plt.close()
    print(f"   → '{cikti}' kaydedildi.\n")


def modeli_egit(df: pd.DataFrame):
    print("3. MODEL EĞİTİLİYOR")

    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    scaler = StandardScaler()
    X.loc[:, "Amount"] = scaler.fit_transform(X[["Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print(f"   Eğitim seti : {len(X_train):,} kayıt")
    print(f"   Test seti   : {len(X_test):,} kayıt\n")

    return model, X_test, y_test


def metrikleri_hesapla(model, X_test, y_test) -> tuple:
    print("4. METRİKLER HESAPLANIYOR")

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrikler = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),
        "F1 Score" : f1_score(y_test, y_pred),
        "ROC-AUC"  : roc_auc_score(y_test, y_pred_prob),
    }

    print("\n  ┌─────────────┬───────────┐")
    print("  │ Metrik      │   Değer   │")
    print("  ├─────────────┼───────────┤")
    for isim, deger in metrikler.items():
        print(f"  │ {isim:<11} │   {deger:.4f}  │")
    print("  └─────────────┴───────────┘")

    fark = metrikler["Accuracy"] - metrikler["F1 Score"]
    print(f"\n  Accuracy − F1 farkı: {fark:.4f}  ← yanılgı!\n")

    return metrikler, y_pred, y_pred_prob


def classification_report_yazdir(y_test, y_pred):
    print("5. CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal (0)", "Dolandırıcılık (1)"]
    ))


def confusion_matrix_gorsellestir(y_test, y_pred, cikti: str = "confusion_matrix.png"):
    print("6. CONFUSION MATRIX")

    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold")

    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Normal (0)", "Dolandırıcılık (1)"],
        yticklabels=["Normal (0)", "Dolandırıcılık (1)"],
        ax=axes[0], linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[0].set_title("Ham Sayılar", fontsize=12)
    axes[0].set_xlabel("Tahmin Edilen", fontsize=11)
    axes[0].set_ylabel("Gerçek", fontsize=11)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=["Normal (0)", "Dolandırıcılık (1)"],
        yticklabels=["Normal (0)", "Dolandırıcılık (1)"],
        ax=axes[1], linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[1].set_title("Normalize (Satır Bazlı)", fontsize=12)
    axes[1].set_xlabel("Tahmin Edilen", fontsize=11)
    axes[1].set_ylabel("Gerçek", fontsize=11)

    etiketler = [["TN", "FP"], ["FN", "TP"]]
    for ax in axes:
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.15, etiketler[i][j],
                        ha="center", va="center", fontsize=9,
                        color="gray", style="italic")

    plt.tight_layout()
    plt.savefig(cikti, bbox_inches="tight")
    plt.close()
    print(f"   → '{cikti}' kaydedildi.\n")


def metrikleri_karsilastir(metrikler: dict, cikti: str = "metrics_comparison.png"):
    print("7. METRİK KARŞILAŞTIRMA GRAFİĞİ OLUŞTURULUYOR")

    isimler = list(metrikler.keys())
    degerler = list(metrikler.values())

    renkler = [
        COLORS["highlight"] if isim == "Accuracy" else COLORS["accent"]
        for isim in isimler
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(isimler, degerler, color=renkler, width=0.55,
                  edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, degerler):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.03,
            f"{val:.4f}",
            ha="center", va="top",
            fontsize=12, fontweight="bold", color="white",
        )

    acc = metrikler["Accuracy"]
    ax.axhline(acc, color=COLORS["highlight"], linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Accuracy eşiği ({acc:.4f})")

    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(
        "Accuracy Yanılgısı: Hangi Metriğe Güvenmeliyiz?",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.set_ylabel("Skor", fontsize=12)
    ax.set_xlabel("Metrik", fontsize=12)
    ax.legend(fontsize=10)

    ax.annotate(
        "Bu yüksek görünüyor\nama yanıltıcı!",
        xy=(0, acc), xytext=(0.6, acc + 0.07),
        fontsize=9, color=COLORS["highlight"],
        arrowprops=dict(arrowstyle="->", color=COLORS["highlight"], lw=1.5),
        ha="center",
    )

    plt.tight_layout()
    plt.savefig(cikti, bbox_inches="tight")
    plt.close()
    print(f"   → '{cikti}' kaydedildi.\n")


def main():
    print("\n" + "=" * 60)
    print("  MAKİNE ÖĞRENMESİNDE ACCURACY YANILGISI")
    print("  Credit Card Fraud Detection — Kaggle")
    print("=" * 60 + "\n")

    # 1. Veriyi yükleyelim
    df = veriyi_yukle("creditcard.csv")

    # 2. Sınıf dağılımını görselleştirelim
    sinif_dagilimini_gorsellestir(df, "images/class_distribution.png")

    # 3. Modeli eğitelim
    model, X_test, y_test = modeli_egit(df)

    # 4. Metrikleri hesaplayalım
    metrikler, y_pred, _ = metrikleri_hesapla(model, X_test, y_test)

    # 5. Classification report
    classification_report_yazdir(y_test, y_pred)

    # 6. Confusion matrix
    confusion_matrix_gorsellestir(y_test, y_pred, "images/confusion_matrix.png")

    # 7. Metrikleri karşılaştıralım
    metrikleri_karsilastir(metrikler, "images/metrics_comparison.png")

    print("  TAMAMLANDI — 3 görsel kaydedildi.")


if __name__ == "__main__":
    main()