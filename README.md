# URL_DETECTION

This repository contains a machine learning model that detects phishing URLs using Logistic Regression. The model is trained on a dataset of phishing and legitimate URLs and provides predictions through a Gradio interface.

### Features
- Uses **Logistic Regression** for classification.
- Extracts features from URLs using **CountVectorizer**.
- Encodes labels using **LabelEncoder**.
- Provides an interactive **Gradio** interface for real-time URL predictions.

### Dataset
The dataset used for training is a CSV file containing URLs labeled as phishing or legitimate.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/url-detection.git
   cd url-detection

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

### Usage

1. Run the script:
   ```bash
   python Url_detection.py
2. A Gradio interface will open in your browser, where you can input a URL to check if it's phishing.

### Dependencies

- `numpy`
- `pandas`
- `gradio`
- `scikit-learn`

Install them using:
```bash
pip install numpy pandas gradio scikit-learn
