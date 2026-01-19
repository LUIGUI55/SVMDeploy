
import base64
import io
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.conf import settings
import os
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def index(request):
    # Load Dataset
    csv_path = os.path.join(settings.BASE_DIR, 'svm_app/data/Phishing.csv')
    df = pd.read_csv(csv_path)

    # Data Preparation (logic from notebook)
    # Target and Features
    # Note: The notebook uses 'URL_Type_obf_Type' as target and splits into train/val/test
    # We will simplify for the dashboard to train on a subset for speed/viz
    
    # Selecting the features used in the plot: domainUrlRatio, domainlength
    
    # 1. Prepare target
    # The notebook does train_val_test_split. Let's do a simple split or use whole data for visualization if it's not too large.
    # Notebook: X_train_reduced = X_train_prepared[["domainUrlRatio", "domainlength"]].copy()
    
    # For this demo, let's just take a sample to keep it fast
    df_sample = df.sample(n=1000, random_state=42) # Adjust size as needed
    
    X = df_sample[["domainUrlRatio", "domainlength"]]
    y = df_sample["URL_Type_obf_Type"]
    
    # Map target to binary for SVM (benign vs phishing) - Notebook logic seems to handle strings but SVC needs numbers or handled by pipeline? 
    # Notebook uses SVC directly on mapped or string data? 
    # Viewing notebook cell 7697: y_train == "phishing" / "benign". 
    # Sklearn SVC usually requires numeric y, or handles strings if LabelEncoder used? 
    # Wait, SVC in sklearn supports any hashable class labels, but usually it's better to encode.
    # Let's verify if we need to encode. It's safer to encode.
    
    # Simple binary mapping for visualization
    # We focus on Benign vs Phishing (or Malicious). Dataset has multiple types?
    # Notebook focuses on "Phishing" vs "Benign" in cell 7697.
    # Let's filter only Benign and Phishing for the clear plot
    
    df_filtered = df[(df["URL_Type_obf_Type"] == 'benign') | (df["URL_Type_obf_Type"] == 'phishing')].copy()
    
    # Reduce size for speed in web request
    if len(df_filtered) > 2000:
        df_filtered = df_filtered.sample(n=2000, random_state=42)

    X_filtered = df_filtered[["domainUrlRatio", "domainlength"]]
    y_filtered = df_filtered["URL_Type_obf_Type"]
    
    # Convert labels to 0/1 for plotting logic convenience (or keep strings)
    # y_numeric = y_filtered.map({'benign': 0, 'phishing': 1})

    # Train SVM
    # Notebook: svm_clf = SVC(kernel="linear", C=50)
    # Notebook uses Pipeline with RobustScaler in Cell 24, but Cell 20 uses just SVC on "X_train_reduced".
    # Cell 7696 plots "svm_clf" which is from Cell 20 (raw SVC).
    # However, unscaled SVM on features with different scales (ratio 0-1 vs length 0-100+) is bad. 
    # But we want to REPLICATE the notebook. Cell 20: svm_clf = SVC(kernel = "linear", C=50) on X_train_reduced.
    # Let's stick to the raw SVC as per Cell 7699 request "plot_svc_decision_boundary(svm_clf, 0, 1)".
    
    svm_clf = SVC(kernel="linear", C=50)
    svm_clf.fit(X_filtered, y_filtered)

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.plot(X_filtered.values[:, 0][y_filtered == "phishing"], X_filtered.values[:, 1][y_filtered == "phishing"], "g^", label="Phishing")
    plt.plot(X_filtered.values[:, 0][y_filtered == "benign"], X_filtered.values[:, 1][y_filtered == "benign"], "bs", label="Benigno")
    
    # Decision Boundary Logic
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    
    # x0 range (domainUrlRatio)
    x0 = np.linspace(0, 1, 200)
    # decision_boundary formula: w0*x0 + w1*x1 + b = 0 => x1 = -w0/w1 * x0 - b/w1
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    
    # Notebook plot style
    # svs = svm_clf.support_vectors_
    # plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA', label="Support Vectors")
    
    plt.plot(x0, decision_boundary, "k-", linewidth=2, label="Límite de Decisión")
    plt.plot(x0, gutter_up, "k--", linewidth=2, label="Margen")
    plt.plot(x0, gutter_down, "k--", linewidth=2)
    
    plt.title(f"Kernel Lineal SVM (C={svm_clf.C})", fontsize=16)
    plt.xlabel("Ratio URL del Dominio", fontsize=12)
    plt.ylabel("Longitud del Dominio", fontsize=12)
    
    # Interactive/Responsive adjustments if any (using static png for now)
    plt.axis([0, 1, -10, 150]) # Adjusted axis slightly to fit data better or use notebook's [0,1, -100, 250]
    # Notebook: plt.axis((0,1, -100, 250))
    plt.axis((0, 1, -100, 250))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plot_data = uri
    plt.close()

    context = {
        'plot_data': plot_data,
        'accuracy': f"{svm_clf.score(X_filtered, y_filtered):.2%}"
    }
    return render(request, 'svm_app/index.html', context)
