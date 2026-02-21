# Food-freshness-detection-system-using-image-processing
Image-based food freshness detection system using OpenCV, texture analysis (GLCM), and computer vision techniques
Food Freshness Detection System

🍎 Food Freshness Detection System

This project is an image-based food freshness analysis system built using Python and OpenCV. It analyzes food images and classifies them as Fresh, Moderately Fresh, or Spoiled based on multiple visual features.

🚀 Features

📷 Analyze multiple images from a folder

🧠 Uses computer vision + image processing

🎯 Detects:

Brightness

Texture (GLCM)

Edge Density

Entropy

Brown Spots (Spoilage indicator)

🖱️ Click on image to get freshness result

📊 Displays detailed analysis values

🛠️ Technologies Used

Python

OpenCV (cv2)

NumPy

Matplotlib

Scikit-image (GLCM texture analysis)

📂 Project Structure
Food-Freshness-Detection/
│── images/              # Input images folder
│── main.py              # Main Python script
│── README.md            # Project documentation
⚙️ How It Works

Loads images from a folder

Extracts features:

HSV Brightness

Texture using GLCM

Edge detection

Entropy calculation

Detects brown spots using HSV masking

Classifies freshness using predefined thresholds

▶️ How to Run
Step 1: Install dependencies
pip install opencv-python numpy matplotlib scikit-image
Step 2: Add images

Create a folder named:

images

Place food images inside it.

Step 3: Run program
python main.py
📸 Output

Displays image grid

Click any image → shows freshness result

Prints detailed analysis in console

📈 Example Output
====== FOOD FRESHNESS ANALYSIS ======
Image Name        : apple.jpg
Mean Brightness   : 135.45
Contrast          : 3.21
Edge Density      : 0.0231
Entropy           : 5.12
Brown Spot Ratio  : 0.0123
Freshness         : FRESH
=====================================
🔮 Future Improvements

Use Machine Learning / Deep Learning

Real-time camera detection

Mobile app integration

Dataset-based training

👩‍💻 Author

Priyanka S
Information Science Student
