# 🌌 NEA Detection Dashboard for Lesedi 1-m Telescope

This project provides a **Streamlit-based dashboard** for visualizing and scoring Near-Earth Asteroid (NEA) detections from the **Lesedi 1-m telescope at SAAO**.  
It integrates the full detection pipeline — from frame alignment to **hybrid CNN + Gradient Boosting** scoring — and outputs ranked asteroid candidates.

---

## Features

- Upload aligned triplet folders or ZIPs directly in the app.  
- Automatically runs the **hybrid CNN + GB** scoring pipeline.  
- Displays top-ranked candidate tracks with confidence scores.  
- Exports RA/Dec positions of the top-3 candidates for MPC submission.  
- Includes ready-made **sample ZIPs** for quick testing:
  - `sample.zip` → **High-score** (good linking example)
  - `sample_2.zip` → **Low-score** (low score linking example)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

2. pip install -r requirements.txt
3. run the dashboard - streamlit run app.py
