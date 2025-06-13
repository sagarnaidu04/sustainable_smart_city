# 🌆 Sustainable Smart City Assistant

An AI-powered web application that supports smart city governance, citizen engagement, and sustainability initiatives. Built using **Gradio** and powered by **IBM Granite LLM**, this assistant provides intelligent tools for policy analysis, public reporting, KPI forecasting, anomaly detection, eco-awareness, and interactive Q&A.

---

## 🚀 Features

### 📝 Policy Search & Summarization
Upload or paste long policy documents and get citizen-friendly summaries using IBM Granite LLM.

### 📣 Citizen Feedback Reporting
Citizens can submit infrastructure issues or service complaints directly through the app.

### 📊 KPI Forecasting
Upload past KPI CSV files (e.g., water usage, traffic volume) and get AI-predicted values for future planning.

### 🌱 Eco Tips Generator
Type keywords like "plastic" or "solar" to get actionable and educational eco-friendly tips.

### 🚨 Anomaly Detection
Upload city metric CSV files with a `value` column to identify outliers or unexpected spikes in usage.

### 💬 Chat Assistant
Ask questions like "How can my city reduce emissions?" and get LLM-powered, actionable advice.

---

## 🧠 Powered By

- **IBM Granite LLM** from Hugging Face: [`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
- **Gradio UI**: Clean, interactive interface using `Blocks` and custom theming
- **Scikit-learn**: For KPI prediction via linear regression
- **Pandas / NumPy**: Data ingestion and processing

---

## 💻 Technologies Used

- `gradio`
- `transformers`
- `torch`
- `pandas`
- `scikit-learn`
- `accelerate`

---

## 📁 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/sagarnaidu04/smart_city_assistant.git
cd smart-city-assistant

# Create virtual environment and install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

> The app will be hosted on `http://localhost:7860/` or a public link if using Colab.

---

## 🌐 Deploy on Hugging Face Spaces

1. Go to https://huggingface.co/spaces
2. Create a new Space → Select SDK: Gradio
3. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Done! Your app is live and shareable.
