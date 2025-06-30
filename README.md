# 🛢️ Segmentazione di Oil Spill in Immagini SAR a Bassa Qualità

**OilSpillSegmentation** è un progetto di deep learning per la rilevazione automatica di sversamenti di petrolio in immagini SAR (Synthetic Aperture Radar) caratterizzate da bassa risoluzione e forte rumore. Il lavoro confronta diverse architetture neurali per la segmentazione semantica e tecniche di preprocessing.

Sviluppato per il corso di **Computer Vision e Deep Learning** presso l'Università Politecnica delle Marche.

---

## 📌 Obiettivi del Progetto

- Segmentare sversamenti di petrolio in immagini SAR ottenute da satelliti **Sentinel-1** e **Palsar**.
- Confrontare architetture deep learning: U-Net, U-Net++, TransUNet e LightMUNet.
- Confrontare tecniche di denoising: Box filter, Gaussian filter, Median filter, Bilateral filter e Fast NL-Means.
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

Prima del training:
- Applicazione del Fast NL-Means all'intero dataset

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
## 📁 Struttura del Progetto

```plaintext
OilSpillSegmentation/           # Working directory
├── emissions/                  # Directory per i risultati del tracciamento delle emissioni
├── logs/                       # Directory per i file di log
├── module_test/                # Directory per i test su predizioni e denoising
├── saves/                      # Directory per i salvataggi dei modelli
├── sos-dataset/                # Directory con il dataset SOS e la sua versione denoised
├── src/                        # Directory con il codice
│   ├── models/                 # Modelli implementati
│   │   ├── __init__.py
│   │   ├── LightMUNet.py       # Versione LightMUNet utilizzata (LightMUNet_4)
│   │   ├── LightMUNet_1.py     # Versione fedele agli SSM, con matrici A, B, C, D e utilizzo di tf.scan
│   │   ├── LightMUNet_2.py     # Come LightMUNet_1, ma modificata per essere più fedele all'articolo
│   │   ├── LightMUNet_3.py     # Versione semplificata con Conv1D e gating, senza tf.scan
│   │   ├── LightMUNet_4.py     # Come LightMUNet_3, ma migliorati gating e generazione della sequenza
│   │   ├── LightMUNet_5.py     # Altra versione fedele agli SSM, con matrici A, B, C, D e utilizzo di tf.scan
│   │   ├── LightMUNet_6.py     # Versione semi-semplificata con scan custom e update dello stato, senza matrici
│   │   ├── LightMUNet_7.py     # Versione semplificata con sliding window al posto dello scan
│   │   ├── TransUNet.py      
│   │   ├── UNet.py
│   │   ├── UNetL.py
│   │   ├── UNetPP.py
│   │   ├── UNetPPL.py
│   │   └── VMUNetV2.py         # Non utilizzata, pesante
│   │
│   ├── utils/                  # Funzioni e classi ausiliarie
│   │   ├── __init__.py
│   │   ├── BatchLoader.py      # Classe per il caricamento dei batch
│   │   ├── CO2Tracker.py       # Classe per il tracciamento delle emissioni di CO2
│   │   ├── DatasetUtils.py     # Classe con metodi di utilità per il dataset, tra cui il preprocessing
│   │   ├── Denoiser.py         # Classe che implementa i filtri
│   │   ├── misc.py             # Funzioni varie
│   │   ├── ModelLoader.py      # Classe per caricare i modelli
│   │   └── SavesManager.py     # Classe per gestire il salvataggio e il caricamento di file
│   │
│   ├── config.py               # Configurazione dei modelli e del dataset
│   ├── dataclass.py            # Interfacce dati
│   ├── main.py                 # Script principale di esecuzione
│   ├── predict.py              # Script per l'inferenza su immagini
│   └── train.py                # Script per l’addestramento dei modelli
│
├── .gitignore
├── README.md                   # Documentazione del progetto
└── requirements.txt            # Dipendenze Python
```


## 🧪 Come Eseguire il Codice

### Entry point
Il blocco `if __name__ == "__main__"`:
- Configura la GPU
- Inizializza il logger
- Esegue la funzione `main()`
- Registra eventuali eccezioni durante l’esecuzione

### 1. Hardware e sistema
Durante l'intero progetto si è fatto uso di una scheda video NVIDIA GeForce RTX 4070 per velocizzare l'addestramento delle reti.
Per poter far uso della GPU, il progetto è stato eseguito in WSL2 con Ubuntu 24.04.2 LTS; in alternativa, è possibile ricorrere a un container Docker.

Il progetto è predisposto all'utilizzo di una sola GPU. Si faccia riferimento a [questa documentazione](https://www.tensorflow.org/guide/keras/distributed_training) per l'addestramento distribuito.
Si consiglia di apportare le modifiche alla funzione `config_gpu()` del file `misc.py`

### 2. Installa Python
La versione Python utilizzata è la 3.10.18.

### 3. Clona il repository
```bash
git clone https://github.com/JoshuaSgariglia/OilSpillSegmentation.git
cd OilSpillSegmentation
```

### 4. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 5. Scarica e preprocessa il dataset
- Scarica il dataset SOS corretto da [questo link](https://drive.google.com/file/d/12grU_EAPbW75eyyHj-U5pOfnwQzm0MFw/view)
- Poni il dataset nella struttura del progetto (directory `sos-dataset/`)
- Applica i filtri di preprocessing usando la funzione `denoise_dataset` nel file `main.py`

Questo passo è necessario se si vuole addestrare e valutare i modelli sul dataset preprocessato, altrimenti può essere saltato.

### 6. Seleziona una funzionalità disponibile nel `main.py`

Il file `main.py` funge da punto di ingresso principale del progetto e permette di attivare, tramite commento/scommento, diverse funzionalità relative all'addestramento, alla valutazione, alla predizione e al monitoraggio.  
Di seguito una panoramica delle opzioni disponibili all’interno della funzione `main()`:

#### Preprocessing

- `denoise_dataset(dataset: DatasetPaths)`  
  Applica il Fast NL-Means all’intero dataset 'dataset'.
  Il nuovo dataset è salvato al percorso `sos-dataset/denoised/`.

- `determine_best_filters(logger: Logger)`  
  Valuta vari filtri di denoising su un subset per determinare la combinazione più efficace.  
  I modelli addestrati e valutati sono salvati al percorso `saves/{nome_dataset}/{nome_modello}`.

---

#### Addestramento e Valutazione

- `train_eval_session(logger: Logger)`  
  Addestra tutti i modelli con diverse configurazioni e ne valuta le performance su validation/test set.
  I modelli addestrati e valutati sono salvati al percorso `saves/{nome_dataset}/{nome_modello}`.

- `determine_best_models(logger: Logger)`  
  Analizza i risultati per identificare i modelli con le migliori metriche.  
  Per ogni dataset i migliori modelli sono salvati al percorso `saves/{nome_dataset}/best_models`.

---

#### Monitoraggio delle Emissioni

- `track_emissions_session(logger: Logger)`  
  Calcola e registra le emissioni di CO₂ associate all’addestramento e alla valutazione tramite la libreria `codecarbon`.
  Le emissioni misurate sono salvate al percorso `emissions/`.
  
---

#### Test su Immagine Singola

- `test_denoising(dataset: DatasetPaths, image_number: int)`  
  Applica varie combinazioni di filtri di denoising a una singola immagine e salva tutti i risultati in `module_test/denoising`.

- `test_prediction(dataset: DatasetPaths, denoised: bool, image_number: int, model_path: str)`  
  Effettua una predizione su una singola immagine usando un modello già addestrato e salva i risultati in 'module_test/prediction`.
  `model_path` è il percorso del file del modello a partire dalla working directory, incluso il nome del file.

---

#### Visualizzazione dei Modelli

- `UNetL.show_model_summary()` → ~1.9M parametri  
- `UNetPPL.show_model_summary()` → ~2.2M parametri  
- `UNet.show_model_summary()` → ~7.8M parametri  
- `UNetPP.show_model_summary()` → ~9.0M parametri  
- `TransUNet.show_model_summary()` → ~100.9M parametri  
- `LightMUNet.show_model_summary()` → ~8.5M parametri  

Visualizza la struttura e la complessità di ciascuna architettura.

---

## 📈 Possibili Estensioni Future

- Utilizzo di funzioni di loss bilanciate (es. focal loss) per migliorare la segmentazione della classe minoritaria.
- Esplorazione di architetture Mamba complete.
- Ottimizzazione per inferenza in tempo reale.

---

## 👨‍🔬 Autori

- **Angelo Kollcaku**
- **Joshua Sgariglia**

> Supervisione: Prof.ssa Lucia Migliorelli, Prof. Alessandro Galdelli, Prof. Adriano Mancini, Prof. Stefano Mereu  
> Università Politecnica delle Marche – A.A. 2024/2025

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza **MIT**.
Consulta il file [LICENSE](LICENSE) per maggiori dettagli.
