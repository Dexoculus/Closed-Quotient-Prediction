# closed-quotient-prediction

This repository contains the implementation of a novel algorithm for predicting Closed Quotient (CQ) values during vocal phonation using GRU-type neural networks, as presented in our paper: [Prediction of Closed Quotient During Vocal Phonation using GRU-type Neural Network with Audio Signals](https://doi.org/10.56977/jicce.2024.22.2.145).

---

## What is Closed Quotient (CQ)?
CQ represents the time ratio for which vocal folds are in contact during voice production. Traditionally, CQ is measured using mechanical or electrical methods, such as inverse filtering or electroglottography.

---

## Why is this research significant?
Our approach eliminates the need for these complex measurement techniques by predicting CQ directly from audio signals. This innovation simplifies the process and makes CQ analysis more accessible, offering a valuable tool for applications like vocal training for professional singers.

---

## How does it work?
We employ GRU-based neural networks to process audio signals, with pitch feature extraction used for pre-processing. By combining GRU architectures with a dense layer for final prediction, our model achieves high accuracy in predicting CQ values, as demonstrated by a low mean squared error in evaluation.

---

## Key Features
This repository includes 4 Deep Neural Network models and 2 Machine Learning models:
- **GRU** (Gated Recurrent Unit)
- **GRU2L** (2 Layers of GRU)
- **BiGRU** (Bidirectional GRU)
- **CNN_GRU** (1D CNN + GRU)
- **Random Forest**
- **XGBoost**

---

## Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/Dexoculus/closed-quotient-prediction.git
cd closed-quotient-prediction
pip install -r requirements.txt
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements
We thank all contributors and collaborators who supported this research.

---

## Citation
If you use this code or research in your work, please cite the paper:
> [Prediction of Closed Quotient During Vocal Phonation using GRU-type Neural Network with Audio Signals](https://doi.org/10.56977/jicce.2024.22.2.145)

```
@article{han2024prediction,
  title={Prediction of Closed Quotient During Vocal Phonation using GRU-type Neural Network with Audio Signals},
  author={Han, Hyeonbin and Lee, Keun Young and Shin, Seong-Yoon and Kim, Yoseup and Jo, Gwanghyun and Park, Jihoon and Kim, Young-Min},
  year={2024},
  publisher={Korea Institute of Information and Communication Engineering}
}
```
