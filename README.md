# 🌿 AI Leaf Disease Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leafdiseaseai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Advanced AI-powered leaf disease detection system with multi-crop support, real-time predictions, and bilingual advisory in English & Hindi.**

## 🚀 Live Demo
**[Try the App →](https://leafdiseaseai-4bglstbxjjusmtynwoyaxy.streamlit.app/)**

## ✨ Key Features

### 🔬 **AI-Powered Detection**
- **MobileNetV2** deep learning architecture
- **Real-time image analysis** with high accuracy
- **Multi-crop support**: Sugarcane, Tomato, Potato, Pepper
- **12+ disease types** detection for sugarcane
- **15+ disease types** for other crops

### 🌍 **Bilingual Support**
- **English & Hindi** interface
- **Text-to-speech** advisory in both languages
- **Localized disease information** and treatments

### 📱 **User-Friendly Interface**
- **Drag & drop** image upload
- **Webcam integration** for instant capture
- **Confidence-based predictions** with top-3 alternatives
- **Interactive feedback system**

### 📊 **Smart Analytics**
- **Real-time dashboard** with prediction analytics
- **Feedback tracking** and model improvement
- **Performance metrics** visualization

## 🎯 Supported Crops & Diseases

### 🌾 **Sugarcane** (12 Diseases)
- Banded Chlorosis, Brown Spot, Brown Rust
- Dried Leaves, Grassy Shoot, Healthy Leaves
- Pokkah Boeng, Sett Rot, Smut
- Viral Disease, Yellow Leaf, Red Stripe

### 🍅 **Other Crops** (15 Diseases)
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, etc.
- **Potato**: Early Blight, Late Blight, Healthy
- **Pepper**: Bacterial Spot, Healthy

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: TensorFlow 2.20.0, MobileNetV2
- **Image Processing**: PIL, OpenCV
- **Audio**: Google Text-to-Speech (gTTS)
- **Data**: Pandas, NumPy
- **Deployment**: Streamlit Cloud

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Tyagideepak108/LeafDiseaseAI.git
cd LeafDiseaseAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```

### 4. Open Browser
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
LeafDiseaseAI/
├── 🎯 app.py                    # Main Streamlit application
├── 🔮 predict.py               # Prediction logic
├── 🧠 models/                  # Trained ML models
│   ├── sugarcane_phase2_best.h5
│   └── other_crops_model_best.h5
├── 📚 sugercane_info.json      # Disease information (Sugarcane)
├── 📚 other_diseases_info.json # Disease information (Other crops)
├── 📋 requirements.txt         # Dependencies
├── 🚫 .gitignore              # Git ignore rules
└── 📖 README.md               # This file
```

## 🎨 Features in Detail

### 🔍 **Smart Prediction System**
- **Confidence threshold**: Shows top-3 predictions when model uncertainty is high
- **Real-time processing**: Instant results with visual feedback
- **Error handling**: Graceful fallbacks for edge cases

### 🗣️ **Audio Advisory**
- **Bilingual TTS**: Disease causes in English & Hindi
- **Treatment guidance**: Organic, chemical, and preventive measures
- **Accessibility**: Audio support for visually impaired users

### 📈 **Analytics Dashboard**
- **Prediction accuracy** tracking
- **User feedback** analysis
- **Disease prevalence** statistics
- **Model performance** metrics

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Deepak Tyagi**
- GitHub: [@Tyagideepak108](https://github.com/Tyagideepak108)
- LinkedIn: [Connect with me](www.linkedin.com/in/tyagi-deepak)

## 🙏 Acknowledgments

- **PlantVillage Dataset** for training data
- **TensorFlow Team** for the amazing framework
- **Streamlit** for the incredible web app framework
- **Open Source Community** for continuous inspiration

## 📞 Support

If you find this project helpful, please ⭐ **star** the repository!

For issues and questions, please [open an issue](https://github.com/Tyagideepak108/LeafDiseaseAI/issues).

---

<div align="center">
  <strong>🌱 Empowering farmers with AI-driven crop health insights 🌱</strong>
</div>