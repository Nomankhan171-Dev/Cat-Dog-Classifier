# Cat & Dog Classifier

A portfolio-ready **Machine Learning + Streamlit + Python** image classification project.

## Classes
- Cat
- Dog

## Features
- Upload pet images
- Image preview
- Cat/Dog prediction
- Confidence score
- Probability bars
- Same dark portfolio theme as the Vehicle Classification App
- Optional trained scikit-learn model
- Demo fallback when no model is present

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Train the model

Create:

```text
dataset/
  Cat/
  Dog/
```

Put labeled images in each folder and run:

```bash
python train_model.py
```

The trained model will be saved as:

```text
model/cat_dog_classifier.pkl
```

## Streamlit deployment

Use:

```text
Main file path: app.py
```

## Note
Without a trained model file, the app runs in demo mode for portfolio presentation.
