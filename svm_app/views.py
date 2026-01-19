
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
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plt.close(fig)
    return uri

def index(request):
    # Load Dataset
    csv_path = os.path.join(settings.BASE_DIR, 'svm_app/data/Phishing.csv')
    df = pd.read_csv(csv_path)

    # Data Preparation
    # Filter Benign vs Phishing
    df_filtered = df[(df["URL_Type_obf_Type"] == 'benign') | (df["URL_Type_obf_Type"] == 'phishing')].copy()
    
    # Stratified Sample to keep classes balanced-ish and fast
    if len(df_filtered) > 1500:
        df_filtered = df_filtered.sample(n=1500, random_state=42)

    X_filtered = df_filtered[["domainUrlRatio", "domainlength"]]
    y_filtered = df_filtered["URL_Type_obf_Type"]
    
    # Train SVM
    svm_clf = SVC(kernel="linear", C=50)
    svm_clf.fit(X_filtered, y_filtered)
    
    # Metrics
    y_pred = svm_clf.predict(X_filtered)
    f1 = f1_score(y_filtered, y_pred, pos_label='phishing')
    cm = confusion_matrix(y_filtered, y_pred, labels=svm_clf.classes_)

    # 1. Decision Boundary Plot
    fig_boundary = plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.plot(X_filtered.values[:, 0][y_filtered == "phishing"], X_filtered.values[:, 1][y_filtered == "phishing"], "g^", label="Phishing")
    plt.plot(X_filtered.values[:, 0][y_filtered == "benign"], X_filtered.values[:, 1][y_filtered == "benign"], "bs", label="Benigno")
    
    # Decision Boundary
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    x0 = np.linspace(0, 1, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    
    plt.plot(x0, decision_boundary, "k-", linewidth=2, label="Límite de Decisión")
    plt.plot(x0, gutter_up, "k--", linewidth=2, label="Margen")
    plt.plot(x0, gutter_down, "k--", linewidth=2)
    
    # Handle User Input (Simulation of "Upload Email")
    user_point = None
    prediction_text = None
    
    if request.method == 'POST':
        try:
            u_ratio = float(request.POST.get('ratio', 0))
            u_length = float(request.POST.get('length', 0))
            
            # Predict
            user_input = np.array([[u_ratio, u_length]])
            pred_class = svm_clf.predict(user_input)[0]
            
            # Highlight on plot
            plt.scatter(u_ratio, u_length, s=300, c='yellow', marker='*', edgecolors='black', label='Tu Caso (Email)', zorder=10)
            
            prediction_text = "Phishing" if pred_class == "phishing" else "Benigno"
            user_point = {'ratio': u_ratio, 'length': u_length, 'prediction': prediction_text}
            
        except ValueError:
            pass

    plt.title(f"Kernel Lineal SVM (C={svm_clf.C})", fontsize=16)
    plt.xlabel("Ratio URL del Dominio", fontsize=12)
    plt.ylabel("Longitud del Dominio", fontsize=12)
    plt.axis([0, 1, -10, 150])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_data = plot_to_base64(fig_boundary)

    # 2. Confusion Matrix Plot
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_clf.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
    ax_cm.set_title("Matriz de Confusión")
    plot_cm = plot_to_base64(fig_cm)

    # 3. Bar Chart (Feature Importance)
    # For linear kernel, weights (coef_) represent feature importance
    fig_bar = plt.figure(figsize=(6, 4))
    features = ["Ratio URL", "Longitud"]
    importance = np.abs(svm_clf.coef_[0]) # formatting for magnitude
    colors = ['#38bdf8', '#818cf8']
    
    plt.bar(features, importance, color=colors)
    plt.title("Importancia de Características (Pesos)")
    plt.ylabel("Magnitud del Peso")
    plt.grid(axis='y', alpha=0.3)
    
    plot_bar = plot_to_base64(fig_bar)

    context = {
        'plot_data': plot_data,
        'plot_cm': plot_cm,
        'plot_bar': plot_bar,
        'accuracy': f"{svm_clf.score(X_filtered, y_filtered):.2%}",
        'f1_score': f"{f1:.2f}",
        'user_point': user_point
    }
    return render(request, 'svm_app/index.html', context)
