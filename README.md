<!-- # EEG Emotion Classification

This project uses EEG spectral features to classify emotional states using machine learning.

## Folder Structure

- `data/`: Contains your dataset (`emotions.csv`)
- `models/`: Trained model will be saved here
- `src/`: Modular source code for each pipeline stage
- `main.py`: Entry script to run the full pipeline
- `requirements.txt`: List of dependencies

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```
   python main.py
   ```

## Dataset
Used dataset: [EEG Brainwave Emotion Dataset](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) -->


# EEG Emotion Classification Pipeline

## Project Overview
This project implements an EEG-based emotion classification pipeline using machine learning. It loads EEG data, preprocesses it, trains multiple classifiers, compares their performance, and visualizes results such as accuracy bar charts and confusion matrices. The best performing model is saved for later predictions.

---

## Features
- Data loading and preprocessing
- Model training with cross-validation (Random Forest, SVM, Gradient Boosting)
- Automatic best model selection and saving
- Visualization of model accuracies and confusion matrix
- Predict new data using the saved model
- Modular and easy to extend

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone <your-repo-URL>
cd <your-repo-folder>
2. Create and activate a virtual environment
On macOS/Linux:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
On Windows:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Dataset
Place your EEG dataset CSV file at:

bash
Copy
Edit
data/emotions.csv
Dataset format:

One column for labels (e.g., NEGATIVE, NEUTRAL, POSITIVE)

Other columns as numeric EEG features

Running the Pipeline
To run the full pipeline (data loading, preprocessing, training, evaluation, visualization):

bash
Copy
Edit
python main.py
This will:

Load and preprocess data

Train models with cross-validation

Display accuracy summary and bar chart

Plot confusion matrix for the best model

Save the best model at models/model.pkl

Predicting New Data
To predict labels on new EEG data using the saved model:

bash
Copy
Edit
python src/predict.py --file data/new_eeg_data.csv
Replace data/new_eeg_data.csv with your new dataset path.

Visualizations
Accuracy Bar Chart: Compares the test accuracies of all trained models.

Confusion Matrix: Shows classification performance of the best model.

(Add screenshots here if possible for better presentation)

Project Structure
bash
Copy
Edit
├── data/
│   └── emotions.csv          # Dataset
├── models/
│   └── model.pkl             # Saved best model
├── src/
│   ├── load_data.py          # Data loading functions
│   ├── preprocess.py         # Preprocessing functions
│   ├── train_models.py       # Model training and selection
│   ├── visualize.py          # Plotting functions
│   └── predict.py            # Prediction script
├── main.py                   # Pipeline entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
Notes
Make sure to NOT commit your venv/ and models/ folders by using .gitignore

You can easily add new models by extending train_models.py

The pipeline uses scikit-learn for ML and matplotlib for visualization

License
(Add license info here if applicable)

Contact
For questions or contributions, please contact:

Your Name – [your.email@example.com]

Happy coding and analyzing EEG data!