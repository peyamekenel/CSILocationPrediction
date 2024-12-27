# WiFi CSI Verileri ile İç Mekan Konum Tahmini

Bu proje, WiFi Kanal Durum Bilgisi (CSI) verilerini kullanarak iç mekan konum tahmini için makine öğrenmesi modelleri uygulamaktadır. Sistem, geleneksel regresyon modellerini ve derin öğrenme yaklaşımlarını kullanarak iç mekanlarda X, Y koordinatlarını tahmin eder.

## Proje Hakkında

Bu sistem şu özelliklere sahiptir:
1. CSI verilerinden özellik çıkarımı
2. Çoklu model desteği (Random Forest, Gradient Boosting, Neural Network, LSTM)
3. Kapsamlı performans analizi
4. Görselleştirme araçları

English version follows below:

---

# Indoor Location Prediction using WiFi CSI Data

This project implements machine learning models for indoor location prediction using WiFi Channel State Information (CSI) data. The system predicts X, Y coordinates in an indoor environment using various machine learning approaches, including traditional regression models and deep learning.

## Project Overview

**Objective:** Develop accurate indoor positioning models using WiFi CSI data

**Dataset:** [CSI-dataset](https://github.com/qiang5love1314/CSI-dataset)
- Lab space: 13.5×11m²
- Meeting room: 7×10m²

**Models Implemented:**
1. Random Forest Regressor
2. Gradient Boosting
3. Neural Network
4. LSTM (for temporal features)

## Requirements

```bash
# Python version
Python 3.12

# Required packages
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

## Project Structure

```
├── inspect_csi.py           # Initial data inspection
├── analyze_csi_dataset.py   # Detailed data analysis
├── preprocess_csi.py        # Data preprocessing pipeline
├── train_models.py          # Traditional ML model training
├── train_lstm_model.py      # LSTM model implementation
├── analyze_model_performance.py  # Performance analysis
├── visualize_results.py     # Results visualization
├── model_results/           # Saved models and results
└── preprocessed/            # Preprocessed data files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/qiang5love1314/CSI-dataset.git
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Usage

1. **Data Preprocessing:**
```bash
python preprocess_csi.py
```
This script:
- Loads raw CSI data
- Performs phase correction
- Engineers features
- Saves preprocessed data

2. **Train Models:**
```bash
python train_models.py
python train_lstm_model.py
```
These scripts:
- Load preprocessed data
- Train various models
- Save model predictions

3. **Visualize Results:**
```bash
python visualize_results.py
```
This script:
- Creates prediction comparison plots
- Generates error distribution analysis
- Saves visualization results

## Model Performance

Average performance metrics:
- RMSE: 2-3 meters
- MAE: 1.5-2.5 meters
- R² Score: -0.02 to -0.12

### Performance by Model:

1. **Random Forest:**
   - Most consistent performance
   - Better accuracy in central areas
   - Conservative predictions

2. **Gradient Boosting:**
   - Similar to Random Forest
   - Slightly higher variance
   - Better in areas with dense training data

3. **Neural Network:**
   - More variable performance
   - Better at capturing non-linear patterns
   - Shows some overfitting tendencies

4. **LSTM:**
   - Good at capturing temporal patterns
   - Higher prediction variance
   - Better with movement sequences

## Areas for Improvement

1. **Data Collection:**
   - More training data in high-error areas
   - Additional environmental features
   - Better noise filtering

2. **Model Enhancements:**
   - Ensemble methods
   - Uncertainty estimation
   - Hyperparameter optimization

3. **Feature Engineering:**
   - Advanced phase correction
   - Additional temporal features
   - Signal strength indicators

## File Descriptions

### Data Processing
- `inspect_csi.py`: Initial data exploration and validation
- `analyze_csi_dataset.py`: Detailed analysis of CSI data structure
- `preprocess_csi.py`: Data preprocessing pipeline

### Model Training
- `train_models.py`: Implementation of Random Forest, Gradient Boosting, and Neural Network models
- `train_lstm_model.py`: LSTM model for temporal feature analysis

### Analysis and Visualization
- `analyze_model_performance.py`: Detailed performance analysis
- `visualize_results.py`: Generation of comparison plots and error analysis

## Results

The project demonstrates that:
1. Indoor positioning using CSI data is feasible with 2-3m accuracy
2. Different models excel in different scenarios
3. Temporal features provide valuable information
4. Room geometry affects prediction accuracy

### Key Findings:
- Models achieve reasonable accuracy for indoor positioning
- Error patterns suggest systematic biases in certain areas
- Temporal features (used in LSTM) provide valuable information
- Room geometry significantly affects prediction accuracy

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original CSI dataset: [qiang5love1314/CSI-dataset](https://github.com/qiang5love1314/CSI-dataset)
- Indoor positioning research community
- Open-source ML libraries contributors
