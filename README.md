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
- **Smart confidence system** with quality-based predictions

### 🌍 **Bilingual Support**
- **English & Hindi** complete interface
- **Text-to-speech** advisory in both languages
- **Localized disease information** and treatments
- **Dynamic language switching** without page refresh

### 📱 **Modern User Interface**
- **Drag & drop** image upload
- **Webcam integration** for instant capture
- **Modern sidebar navigation** with smooth animations
- **Interactive feedback system**
- **Professional gradient design**

### 📊 **Advanced Analytics Dashboard**
- **Enhanced analytics** with 4 comprehensive tabs
- **Real-time metrics**: Total predictions, accuracy, confidence
- **Trend analysis**: Daily patterns, weekly usage
- **Crop-wise analytics** and disease severity mapping
- **Data export** functionality (CSV download)
- **Interactive filtering** by date and crop type

### 📈 **Smart Features**
- **Prediction history** tracking (last 10 predictions)
- **Batch processing** for multiple images
- **Visual progress indicators**
- **Comprehensive feedback system**

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

<<<<<<< HEAD
- **Frontend**: Streamlit with custom CSS styling
=======
- **Frontend**: Streamlit,CSS
>>>>>>> e2ea4046ab56d5aa40fb57c44c6f5e95fbc66d1b
- **AI/ML**: TensorFlow 2.20.0, MobileNetV2
- **Image Processing**: PIL, OpenCV
- **Audio**: Google Text-to-Speech (gTTS)
- **Data Analytics**: Pandas, NumPy, Plotly
- **Visualization**: Interactive charts and graphs
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
├── 🎨 style.css                # Custom CSS styling
├── 🧠 models/                  # Trained ML models
│   ├── sugarcane_phase2_best.h5
│   └── other_crops_model_best.h5
├── 📚 sugercane_info.json      # Disease information (Sugarcane)
├── 📚 other_diseases_info.json # Disease information (Other crops)
├── 📊 feedback.csv             # User feedback data
├── 📁 feedback_images/         # Stored prediction images
├── 📋 requirements.txt         # Dependencies
├── 🚫 .gitignore              # Git ignore rules
└── 📖 README.md               # This file
```

## 🎨 Features in Detail

### 🔍 **Smart Prediction System**
- **Confidence-based results**: Clear predictions with accuracy scores
- **Real-time processing**: Instant results with visual feedback
- **Error handling**: Graceful fallbacks for edge cases
- **Model validation**: Automatic compatibility checking

### 🗣️ **Audio Advisory**
- **Bilingual TTS**: Disease causes in English & Hindi
- **Treatment guidance**: Organic, chemical, and preventive measures
- **Accessibility**: Audio support for visually impaired users
- **Interactive audio controls**

### 📊 **Enhanced Analytics Dashboard**
- **4 Comprehensive Tabs**:
  - 📈 **Trends**: Daily predictions, confidence distribution, weekly patterns
  - 🌾 **Crops**: Crop-wise analysis and accuracy metrics
  - 🦠 **Diseases**: Top diseases and severity mapping
  - 📊 **Details**: Advanced filtering and data export
- **Real-time Metrics**: Live statistics and performance tracking
- **Interactive Visualizations**: Charts, graphs, and data insights
- **Export Functionality**: Download analytics as CSV

### 📱 **Modern Navigation**
- **4 Main Pages**:
  - 🏠 **Home**: Single image prediction
  - 📊 **Dashboard**: Advanced analytics
  - 📜 **History**: Previous predictions (last 10)
  - 📋 **Batch Upload**: Multiple image processing
- **Smooth Animations**: Gradient hover effects
- **Responsive Design**: Works on all devices

### 🎯 **Batch Processing**
- **Multiple Image Upload**: Process several images at once
- **Progress Tracking**: Real-time processing status
- **Batch Results**: Organized display with success metrics
- **Error Handling**: Individual image error management

## 🆕 Recent Updates (Latest Version)

### ✨ **New Features Added**
- 📊 **Enhanced Analytics Dashboard** with 4 comprehensive tabs
- 📜 **Prediction History** tracking with visual cards
- 📋 **Batch Upload** for multiple image processing
- 🎨 **Modern UI/UX** with custom CSS and animations
- 🌐 **Complete Bilingual Support** (English & Hindi)
- 📈 **Advanced Data Visualization** with interactive charts
- 💾 **Data Export** functionality (CSV download)
- 🔄 **Smart Navigation** with persistent state management

### 🛠️ **Technical Improvements**
- Modular code architecture
- Enhanced error handling
- Better data management
- Responsive design optimization
- Performance improvements

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
