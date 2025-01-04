# Tennis Video to Text (TennisVideo2Text)

This project explores a series of experiments aimed at analyzing tennis videos. The ultimate goal is to classify player actions (e.g., **diving** vs. **walking**) and generate meaningful insights by processing and analyzing video files. The repository is organized into stages, each representing a critical step or experiment in the video analysis pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
5. [Stages and Experiments](#stages-and-experiments)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

This project consists of a modular set of experiments designed to process and analyze tennis videos. Each experiment focuses on a specific component, such as player detection, ball tracking, or feature extraction. Due to the proprietary nature of the dataset, video files are **not included** in this repository. Users must provide their own data to replicate the experiments.

---

## Features

- **Modular Pipeline**:
  - Organized into independent stages for flexibility and experimentation.
- **Comprehensive Analysis**:
  - Includes court detection, player tracking, action classification, and statistical analysis.
- **Custom Feature Extraction**:
  - Supports experiments with HOG features, SVM training, clustering, and optical flow.
- **Visualization**:
  - Generates visual insights into player movements and detected patterns.

---

## Project Structure

```plaintext
TennisVideo2Text/
├── experiments/                   # Directory for all experimental stages
│   ├── stage1_court_detection/    # Detect the tennis court in video frames
│   ├── stage2_player_detection/   # Detect players in the video
│   ├── stage3_background_subtraction_cv3/  # Background subtraction
│   ├── stage4_hough_transform_probabilistic/  # Line detection using Hough Transform
│   ├── stage5_ball_tracking/      # Track the ball movement
│   ├── stage6_lucas_kanade/       # Optical flow using Lucas-Kanade
│   ├── stage7_people_detection/   # Detect and label people using advanced methods
│   ├── stage8_hog_of_images_visualization/  # Visualize HOG features
│   ├── stage9_train_svm_with_hog_features/  # Train SVM using HOG features
│   ├── stage10_svm_accuracy/      # Evaluate SVM accuracy
│   ├── stage11_dense_optical_flow/ # Dense optical flow experiments
│   ├── stage12_player_detection_in_images/  # Player detection using alternative methods
│   ├── stage13_grouping_generated_features/  # Group and analyze features
│   ├── stage14_clustering_kmeans/  # K-means clustering for features
│   ├── stage14_video_cropping/    # Crop video frames for experiments
│   ├── stage15_histogram_generation/ # Generate histograms from features
│   ├── stage16_variance_for_svm_kernel/  # Analyze SVM kernel variance
│   ├── stage17_svm_training/      # Final SVM training and classification
├── histogram_files/               # Stores histogram data for analysis
│   ├── 1/                         # Folder for Diving histograms
│   ├── 2/                         # Folder for Walking histograms
└── README.md                      # Project documentation
```

---

## Getting Started

### Prerequisites

- **Python**: Version 3.8 or higher
- **Required Libraries**:
  - `numpy`: For numerical operations
  - `opencv-python`: For video and image processing
  - `matplotlib`: For visualization
  - `scikit-learn`: For machine learning experiments

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TennisVideo2Text.git
   cd TennisVideo2Text
   ```

---

## Stages and Experiments

This project is divided into **stages**, each focusing on a critical aspect of video analysis:

1. **Stage 1: Court Detection**  
   Detect and crop the tennis court area from video frames.
   
2. **Stage 2: Player Detection**  
   Use image processing techniques to detect players.

3. **Stage 3: Background Subtraction**  
   Extract foreground elements by removing background noise.

4. **Stage 4: Hough Transform**  
   Detect court lines using probabilistic Hough transform.

5. **Stage 5: Ball Tracking**  
   Implement tracking algorithms to follow the ball's movement.

6. **Stage 6: Lucas-Kanade Optical Flow**  
   Use Lucas-Kanade to analyze motion vectors.

7. **Stage 7: People Detection**  
   Detect people using advanced detection algorithms, such as HOG features and pre-trained models.  
   Outputs bounding boxes and classifications for individuals in video frames.

8. **Stage 8-10: HOG Features and SVM**  
   Train an SVM model using Histogram of Oriented Gradients (HOG) features for classification.

9. **Stage 13-14: Clustering and Feature Grouping**  
   Apply clustering algorithms like K-means to group extracted features.

10. **Stage 17: SVM Training**  
    Finalize SVM training and validate its performance.

Each stage folder includes its specific scripts and utility files.

---


## Histogram Label Classification

Histograms of extracted features are used to classify actions as **Diving** or **Walking**. The classification is handled by a trained SVM model:

```python
if clf.predict([data])[0] == 1:
    print("Diving")
else:
    print("Walking")
```

- **Input**: `data` represents the histogram of features extracted from a video frame or sequence.
- **Output**:
  - `Diving`: If the model predicts the class label `1`.
  - `Walking`: If the model predicts the class label `0`.

---

## Contributing

Contributions are welcome! You can contribute by:

- Adding new experimental workflows.
- Improving the existing processing pipelines.
- Proposing enhancements for classification models.

---

## License

This project is licensed under the [MIT License](LICENSE).
