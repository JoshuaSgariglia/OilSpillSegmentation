# 🛢️ Segmentazione di Oil Spill in Immagini SAR a Bassa Qualità

**OilSpillSegmentation** è un progetto di deep learning per la rilevazione automatica di sversamenti di petrolio in immagini SAR (Synthetic Aperture Radar) caratterizzate da bassa risoluzione e forte rumore. Il lavoro confronta diverse architetture neurali per la segmentazione semantica e tecniche di preprocessing.

Sviluppato per il corso di **Computer Vision e Deep Learning** presso l'Università Politecnica delle Marche.

---

## 📌 Obiettivi del Progetto

- Segmentare sversamenti di petrolio in immagini SAR ottenute da satelliti **Sentinel-1** e **Palsar**.
- Confrontare architetture deep learning come U-Net, U-Net++, TransUNet e LightMUNet.
- Migliorare la qualità delle immagini tramite tecniche di denoising.
- Monitorare le prestazioni in termini di accuratezza, tempo e impatto ambientale (emissioni di CO₂).

---

## 📂 Dataset

Il progetto utilizza il **Deep-SAR Oil Spill (SOS Dataset - versione corretta, 2023)**:

- Circa 8.000 immagini SAR (dimensioni 256×256 pixel) da Palsar e Sentinel-1A.
- Suddivise in **80% training**, **10% validation**, **10% test**.
- Maschere binarie (bianco = oil spill, nero = background).
- Immagini in scala di grigi su 3 canali identici (PNG).

---

## 🧼 Preprocessing

Per ridurre il rumore speckle tipico delle immagini SAR, è stata adottata una pipeline di denoising:

1. **Fast NL-Means**
2. **Filtro Gaussiano**

Durante il training:
- Riduzione dei canali da 3 a 1
- Normalizzazione delle immagini e binarizzazione delle maschere
- Applicazione del filtro gaussiano a runtime

---

## 🧠 Architetture Utilizzate

Tutte le reti sono implementate in **TensorFlow** e **Keras**.

| Modello          | Descrizione |
|------------------|-------------|
| `UNet`           | Architettura base encoder-decoder con skip connections |
| `UNet++`         | Variante avanzata con connessioni nidificate |
| `UNetL`          | Versione leggera della UNet |
| `UNet++L`        | Versione leggera della UNet++ |
| `TransUNet`      | Ibrido CNN + Transformer |
| `LightMUNet`     | Utilizza Mamba State Space Models invece dei Transformer |

> 🔍 Il modello TransUNet è stato preso da [TransUNet-tf](https://github.com/awsaf49/TransUNet-tf).

---

## ⚙️ Configurazione dell’Addestramento

- **Loss function**: Binary Cross-Entropy (BCE)
- **Ottimizzatore**: Adam
- **Epoche**: 50 per UNet e UNetL, 40 per le altre reti
- **Learning Rate**: dinamico con decadimento basato su Plateau
- **Metriche**: Accuracy, Precision, Recall, F1-score, IoU

---

## 📊 Risultati

### Dataset Palsar

| Modello                   | mF1-score | mIoU  |
|---------------------------|-----------|-------|
| UNet++                    | **88,8%** | **80,6%** |
| TransUNet (preaddestrata) | 88,6%     | 80,4% |
| UNet                      | 88,6%     | 80,4% |

### Dataset Sentinel

| Modello                   | mF1-score | mIoU  |
|---------------------------|-----------|-------|
| TransUNet (preaddestrata) | **88,2%** | **79,1%** |
| UNet++                    | 87,9%     | 78,6% |
| UNet                      | 87,7%     | 78,3% |

> ✅ Le architetture UNet++ e TransUNet preaddestrata hanno fornito le migliori prestazioni complessive.

---

## 🌱 Impatto Ambientale

Le emissioni di CO₂ sono state monitorate con la libreria [`codecarbon`](https://mlco2.github.io/codecarbon/):

- **Emissioni totali stimate**: ~4.56 kg CO₂eq
- **Modello più energivoro**: LightMUNet

---

## 🧪 Come Eseguire il Codice

### 1. Clona il repository
```bash
git clone https://github.com/JoshuaSgariglia/OilSpillSegmentation.git
cd OilSpillSegmentation
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Scarica e preprocessa il dataset
- Scarica il dataset SOS corretto da [questo link](https://drive.google.com/file/d/12grU_EAPbW75eyyHj-U5pOfnwQzm0MFw/view)
- Applica i filtri di preprocessing usando gli script nella cartella `preprocessing/`

### 4. Addestra un modello
```bash
python train.py --model unet++ --dataset palsar
```

### 5. Valuta un modello addestrato
```bash
python evaluate.py --model-path saved_models/unet++_palsar.h5
```

---

## 📈 Possibili Estensioni Future

- Utilizzo di funzioni di loss bilanciate (es. focal loss) per migliorare la segmentazione della classe minoritaria.
- Esplorazione di architetture Mamba complete.
- Ottimizzazione per inferenza in tempo reale.
- Riduzione delle emissioni tramite pruning o quantizzazione.

---

## 👨‍🔬 Autori

- **Joshua Sgariglia**
- **Angelo Kollcaku**

> Supervisione: Prof.ssa Lucia Migliorelli, Prof. Alessandro Galdelli, Prof. Adriano Mancini, Prof. Stefano Mereu  
> Università Politecnica delle Marche – A.A. 2024/2025

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza **MIT**.  
Consulta il file [LICENSE](LICENSE) per maggiori dettagli.
