import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import graycomatrix, graycoprops

# ==============================
# FUNCTION: FRESHNESS ANALYSIS
# ==============================
def analyze_freshness(image):

    image = cv2.resize(image, (300, 300))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Color Features
    mean_value = np.mean(hsv[:, :, 2])
    std_value = np.std(hsv[:, :, 2])

    # Texture (GLCM)
    glcm = graycomatrix(
        blur,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Edge Density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # Entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    # Brown Spot Detection
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(brown_mask > 0) / brown_mask.size

    # Decision
    if (
        mean_value > 120 and
        contrast < 5 and
        edge_density < 0.05 and
        entropy < 6 and
        brown_ratio < 0.05
    ):
        freshness = "FRESH"
        color = (0, 255, 0)

    elif (
        mean_value > 90 and
        contrast < 10 and
        brown_ratio < 0.15
    ):
        freshness = "MODERATELY FRESH"
        color = (0, 255, 255)

    else:
        freshness = "SPOILED"
        color = (0, 0, 255)

    return freshness, color, mean_value, contrast, edge_density, entropy, brown_ratio


# ==============================
# LOAD MULTIPLE IMAGES
# ==============================

folder_path = "images"   # ✅ Put your folder name here (NOT images.jpg)

image_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
]

images = []
for file in image_files:
    img = cv2.imread(os.path.join(folder_path, file))
    if img is not None:
        images.append((file, img))


# ==============================
# DISPLAY GRID
# ==============================
cols = 3
rows = int(np.ceil(len(images) / cols))

fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
axes = axes.flatten()

for i, (name, img) in enumerate(images):
    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(name)
    axes[i].axis("off")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")


# ==============================
# CLICK EVENT
# ==============================
def on_click(event):

    for i, ax in enumerate(axes):
        if ax == event.inaxes and i < len(images):

            name, img = images[i]

            result = analyze_freshness(img)
            freshness, color, mv, con, ed, ent, br = result

            output = img.copy()
            cv2.putText(
                output,
                f"{freshness}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            print("\n====== FOOD FRESHNESS ANALYSIS ======")
            print(f"Image Name        : {name}")
            print(f"Mean Brightness   : {mv:.2f}")
            print(f"Contrast          : {con:.2f}")
            print(f"Edge Density      : {ed:.4f}")
            print(f"Entropy           : {ent:.2f}")
            print(f"Brown Spot Ratio  : {br:.4f}")
            print(f"Freshness         : {freshness}")
            print("=====================================")

            plt.figure(figsize=(4, 4))
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.title(f"{name} → {freshness}")
            plt.axis("off")
            plt.show()


fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()