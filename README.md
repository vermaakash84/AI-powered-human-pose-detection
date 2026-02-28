# 🧍 AI-Powered Human Pose Detection Web App

A web-based Human Pose Detection system built using MediaPipe Tasks API and Streamlit.

This application allows users to upload a video (.mp4 / .mov), detect full-body pose landmarks (33 keypoints), draw skeleton connections, and download the processed output video.

---

## 🚀 Live Demo

👉 (Add your Streamlit deployment link here)

---

## 🎯 Features

- ✅ Upload MP4 / MOV videos
- ✅ Full-body pose detection (33 landmarks)
- ✅ Skeleton visualization
- ✅ Landmark drawing
- ✅ Download processed video
- ✅ Clean web interface
- ✅ Deployable on Streamlit Cloud

---

## 🧠 Tech Stack

- Python
- MediaPipe (Pose Landmarker – FULL model)
- OpenCV
- Streamlit
- NumPy

---

## 📌 How It Works

1. User uploads a video
2. Each frame is processed using MediaPipe PoseLandmarker
3. 33 body landmarks are detected
4. Skeleton connections are drawn
5. Output video is generated
6. User can preview and download the result

---

## 📂 Project Structure

```
pose-web-app/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation (Local Setup)

Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

---

## 🌍 Deployment

This project can be deployed easily using:

- Streamlit Cloud
- Render
- Railway

Steps for Streamlit Cloud:

1. Push repository to GitHub
2. Visit https://streamlit.io/cloud
3. Connect repository
4. Deploy app

---

## 🎥 Sample Output

(Add screenshots or GIF here for better engagement)

---

## 📈 Future Enhancements

- Joint angle calculation (Knee, Elbow)
- Gym repetition counter
- Multi-person pose detection
- Pose classification (Standing, Sitting, Squat, etc.)
- Real-time webcam support
- Analytics dashboard integration

---

## 👤 Author

Akash Verma  
MBA – Analytics & Data Science  
Passionate about Computer Vision & Applied AI

---

## ⭐ If you found this useful

Give this repository a star ⭐
