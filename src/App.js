import React from "react";
import "./App.css";

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const offset = 70; // sticky header height
  const top = el.getBoundingClientRect().top + window.scrollY - offset;
  window.scrollTo({ top, behavior: "smooth" });
}

function App() {
  return (
    <div className="page">
      {/* Top Nav */}
      <header className="nav">
        <div className="nav-title">
          RSNA Intracranial Aneurysm Detection · AI/ML Project
        </div>
        <nav className="nav-links">
          <button className="chip" onClick={() => scrollToSection("overview")}>
            Overview
          </button>
          <button
            className="chip"
            onClick={() => scrollToSection("environment")}
          >
            Environment
          </button>
          <button
            className="chip"
            onClick={() => scrollToSection("data-prep")}
          >
            Data Prep
          </button>
          <button
            className="chip"
            onClick={() => scrollToSection("train-models")}
          >
            Training (Exp0–Exp5)
          </button>
          <button className="chip" onClick={() => scrollToSection("final")}>
            Final Ensemble
          </button>
          <button className="chip" onClick={() => scrollToSection("demo")}>
            Demo / Test
          </button>
        </nav>
      </header>

      {/* Overview / Hero */}
      <section id="overview" className="hero">
        <div className="hero-main">
          <div className="badge">
            <span className="dot" />
            RSNA – Intracranial Aneurysm Detection
          </div>
          <h1 className="hero-title">
            RSNA-Intracranial-Aneurysm-Detection – Project Summary
          </h1>
          <p className="hero-subtitle">
            This project implements an end-to-end deep learning pipeline for
            intracranial aneurysm detection on TOF-MRA series. It covers
            environment setup, data preparation, detection and classification
            experiments (Exp0–Exp5), pseudo-labeling using external datasets,
            and a final ensemble model.
          </p>

          <div className="pill-row">
            <span className="pill">Python · PyTorch</span>
            <span className="pill">Medical Imaging (TOF-MRA)</span>
            <span className="pill">YOLOv11 / YOLOv5</span>
            <span className="pill">ViT Large · EVA Large</span>
            <span className="pill">MIT-B4 FPN</span>
          </div>

          <div className="hero-metrics">
            <div className="metric-card">
              <div className="metric-label">Dataset</div>
              <div className="metric-value">RSNA IAD</div>
              <div className="metric-note">+ Lausanne & Royal Brisbane</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Task</div>
              <div className="metric-value">Binary Classification</div>
              <div className="metric-note">Aneurysm Present / Absent</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Architecture</div>
              <div className="metric-value">Multi-Stage Pipeline</div>
              <div className="metric-note">Exp0 → Exp5 + Ensemble</div>
            </div>
          </div>
        </div>

        <aside className="hero-side">
          <div className="card">
            <h3 className="card-title">High-level Idea</h3>
            <p className="card-body">
              <strong>Goal:</strong> Detect intracranial aneurysms by combining
              slice-level detection, brain cropping, classification models, and
              pseudo-labeled external datasets.
            </p>
            <ul>
              <li>3-channel slice inputs: (t-1, t, t+1)</li>
              <li>YOLOv11 for aneurysm localization (Exp0)</li>
              <li>YOLOv5 brain detector for cropping (Exp1)</li>
              <li>ViT-Large & EVA-Large classifiers (Exp2, Exp4)</li>
              <li>MIT-B4 FPN multi-task models (Exp3, Exp5)</li>
              <li>Final ensemble of 6 models</li>
            </ul>
          </div>
        </aside>
      </section>

      {/* 1. Environment */}
      <section id="environment">
        <div className="section-header">
          <h2 className="section-title">1. Environment</h2>
          <p className="section-tagline">Base software & dependencies</p>
        </div>

        <div className="two-col">
          <div className="panel">
            <div className="panel-heading">1.1 System Setup</div>
            <ul>
              <li>Ubuntu 22.04 LTS</li>
              <li>CUDA 12.1</li>
              <li>Nvidia Driver Version: 560.35.03</li>
              <li>Python 3.11.13</li>
            </ul>
          </div>

          <div className="panel">
            <div className="panel-heading">1.2 Conda & PyTorch</div>
            <p className="panel-text">
              Create a virtual environment and install PyTorch with CUDA 12.1:
            </p>
            <div className="code-block">
{`conda create -n venv python=3.11.13
conda activate venv
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt`}
            </div>
          </div>
        </div>
      </section>

      {/* 2. Data preparation */}
      <section id="data-prep">
        <div className="section-header">
          <h2 className="section-title">2. Data Preparation</h2>
          <p className="section-tagline">
            Download datasets and convert DICOMs to 3-channel slice images
          </p>
        </div>

        <div className="two-col">
          {/* 2.1 Download data */}
          <div className="panel">
            <div className="panel-heading">2.1 Download Data</div>
            <p className="panel-text">
              Download and organize datasets into the <code>./dataset</code>{" "}
              folder:
            </p>
            <ul>
              <li>
                Download{" "}
                <a
                  href="https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/data"
                  target="_blank"
                  rel="noreferrer"
                >
                  RSNA competition dataset
                </a>{" "}
                and extract to <code>./dataset</code>.
              </li>
              <li>
                Download external datasets{" "}
                <a
                  href="https://openneuro.org/datasets/ds003949/versions/1.0.1"
                  target="_blank"
                  rel="noreferrer"
                >
                  Lausanne_TOFMRA
                </a>{" "}
                and{" "}
                <a
                  href="https://openneuro.org/datasets/ds005096/versions/1.0.3"
                  target="_blank"
                  rel="noreferrer"
                >
                  Royal_Brisbane_TOFMRA
                </a>{" "}
                to <code>./dataset/external</code>.
              </li>
            </ul>
            <div className="code-block">
{`cd dataset/external
sudo snap install aws-cli --classic
aws s3 sync --no-sign-request s3://openneuro.org/ds003949 Lausanne_TOFMRA
aws s3 sync --no-sign-request s3://openneuro.org/ds005096 Royal_Brisbane_TOFMRA`}
            </div>
          </div>

          {/* Dataset structure */}
          <div className="panel">
            <div className="panel-heading">2.1 Dataset Structure</div>
            <div className="code-block">
{`dataset
├── series
│   ├── SeriesInstanceUID*
│   │   └── *.dcm
├── train.csv
├── train_localizers.csv
├── train_kfold.csv                 # custom k-folds
├── kaggle_evaluation
└── external
    ├── Lausanne_TOFMRA
    │   ├── derivatives
    │   └── sub-xxx
    └── Royal_Brisbane_TOFMRA
        ├── derivatives
        └── sub-xxx`}
            </div>
          </div>
        </div>

        {/* 2.2 Prepare images & labels */}
        <div className="panel margin-top">
          <div className="panel-heading">2.2 Prepare Images and Labels</div>
          <p className="panel-text">
            Each image has 3 channels corresponding to slices{" "}
            <code>(t-1, t, t+1)</code>.
          </p>
          <p className="panel-text">Run the following scripts:</p>
          <div className="code-block">
{`cd src/prepare
python dicom2image_slice_level.py
python dicom2image_lausanne.py
python dicom2image_royal_brisbane.py
python prepare_label_slice_level.py`}
          </div>
        </div>
      </section>

      {/* 3. Train models (Exp0–Exp5) */}
      <section id="train-models">
        <div className="section-header">
          <h2 className="section-title">3. Train Models</h2>
          <p className="section-tagline">
            Experiments Exp0 to Exp5 (run sequentially)
          </p>
        </div>

        <div className="exp-grid">
          {/* Left column: Exp0, Exp1, Exp2 */}
          <div>
            {/* 3.1 Exp0 */}
            <div className="exp-card">
              <div className="exp-title">
                3.1 Exp0 – Aneurysm Detection (YOLOv11)
              </div>
              <div className="exp-tag">
                Combine classification + localization labels
              </div>
              <p className="exp-text">
                Using only classification labels (<code>train.csv</code>) is less
                effective than combining with localization labels (
                <code>train_localizers.csv</code>). For each aneurysm centroid
                from <code>train_localizers.csv</code>, search within ±10
                neighboring slices and manually assign bounding boxes using{" "}
                <a
                  href="https://github.com/HumanSignal/labelImg"
                  target="_blank"
                  rel="noreferrer"
                >
                  LabelImg
                </a>
                .
              </p>

              <p className="note">
                This process does not require deep medical knowledge because the
                aneurysm centroids are already provided.
              </p>

              <div className="code-block">
{`cd src/exp0_aneurysm_det
python prepare_label.py
python train.py --fold 0 && python train.py --fold 1 && python train.py --fold 2 && python train.py --fold 3 && python train.py --fold 4`}
              </div>

              <table className="metric-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Fold 0</th>
                    <th>Fold 1</th>
                    <th>Fold 2</th>
                    <th>Fold 3</th>
                    <th>Fold 4</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>mAP50</td>
                    <td>0.705</td>
                    <td>0.647</td>
                    <td>0.766</td>
                    <td>0.702</td>
                    <td>0.691</td>
                  </tr>
                  <tr>
                    <td>mAP50-95</td>
                    <td>0.460</td>
                    <td>0.429</td>
                    <td>0.504</td>
                    <td>0.482</td>
                    <td>0.449</td>
                  </tr>
                </tbody>
              </table>

              <p className="note">
                The 5 trained YOLOv11 models are later used to generate
                pseudo-labels for external data in experiments 3 and 4.
              </p>
            </div>

            {/* 3.2 Exp1 */}
            <div className="exp-card">
              <div className="exp-title">
                3.2 Exp1 – Brain Detection (YOLOv5 v7.0)
              </div>
              <div className="exp-tag">Brain bounding box and cropping</div>
              <p className="exp-text">
                For each <code>SeriesInstanceUID</code>:
              </p>
              <ul>
                <li>Average all slices into a single image.</li>
                <li>
                  Manually annotate brain bounding boxes as two classes:
                  <strong> brain</strong> (axial view) and{" "}
                  <strong>abnormal</strong> (other views).
                </li>
                <li>
                  Crop each slice based on the predicted bounding box from this
                  model.
                </li>
              </ul>
              <p className="note">
                This reduces background noise (e.g., lung regions) and improves
                model accuracy by about <strong>0.03–0.05</strong>.
              </p>

              <div className="code-block">
{`cd src/exp1_brain_det_yolov5_7.0
python prepare_label.py
python train.py --cfg models/yolov5n.yaml --weights yolov5n.pt --data data/brain_det.yaml --batch-size 512 --img-size 640 --device 0 --epochs 150 --name yolov5n_640 --project checkpoints
python python predict_external_dataset.py   # brain bbox on external datasets`}
              </div>

              <table className="metric-table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>P</th>
                    <th>R</th>
                    <th>mAP50</th>
                    <th>mAP50-95</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>all</td>
                    <td>0.997</td>
                    <td>1</td>
                    <td>0.995</td>
                    <td>0.948</td>
                  </tr>
                  <tr>
                    <td>brain</td>
                    <td>0.999</td>
                    <td>1</td>
                    <td>0.995</td>
                    <td>0.991</td>
                  </tr>
                  <tr>
                    <td>abnormal</td>
                    <td>0.995</td>
                    <td>1</td>
                    <td>0.995</td>
                    <td>0.906</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* 3.3 Exp2 */}
            <div className="exp-card">
              <div className="exp-title">
                3.3 Exp2 – Classification Models (ViT Large & EVA Large)
              </div>
              <div className="exp-tag">Trained on RSNA dataset</div>
              <p className="exp-text">
                Two classification models are trained on the RSNA dataset:
                <strong> ViT Large 384</strong> and{" "}
                <strong>EVA Large 384</strong>.
              </p>
              <div className="code-block">
{`cd src/exp2_cls

# For label cleaning in Exp4 and Exp5
python train_5folds.py --cfg configs/vit_large_384.yaml
python eval_5folds.py  --cfg configs/vit_large_384.yaml

python train_5folds.py --cfg configs/eva_large_384.yaml
python eval_5folds.py  --cfg configs/eva_large_384.yaml

# Final single models
python train.py --cfg configs/vit_large_384.yaml
python train.py --cfg configs/eva_large_384.yaml

# Predict on external datasets for pseudo-labeling
python predict_external_dataset.py --cfg configs/vit_large_384.yaml`}
              </div>

              <table className="metric-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>OOF (Weighted AUC)</th>
                    <th>OOF + crop 0.75</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>ViT Large 384</td>
                    <td>0.8491</td>
                    <td>0.8503</td>
                  </tr>
                  <tr>
                    <td>EVA Large 384</td>
                    <td>0.8486</td>
                    <td>0.8551</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Right column: Exp3, cleaning, Exp4, Exp5 */}
          <div>
            {/* 3.4 Exp3 */}
            <div className="exp-card">
              <div className="exp-title">
                3.4 Exp3 – Multi-task Classification + Segmentation
              </div>
              <div className="exp-tag">
                MIT-B4 FPN (image size 384, RSNA dataset)
              </div>
              <p className="exp-text">
                A single model (MIT-B4 FPN) is trained as a multi-task network:
              </p>
              <ul>
                <li>Classification head: aneurysm present / absent</li>
                <li>Segmentation head: auxiliary aneurysm mask</li>
              </ul>
              <p className="note">
                For the final submission, only the classification output is
                used. The auxiliary segmentation helps improve feature learning.
              </p>
              <div className="code-block">
{`cd src/exp3_aux
python train.py --cfg configs/mit_b4_fpn_384.yaml
python eval.py  --cfg configs/mit_b4_fpn_384.yaml
python predict_external_dataset.py --cfg configs/mit_b4_fpn_384.yaml`}
              </div>

              <table className="metric-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>OOF</th>
                    <th>OOF + crop 0.75</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>MIT-B4 FPN 384</td>
                    <td>0.8469</td>
                    <td>0.8549</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* 3.5 Cleaning + pseudo */}
            <div className="exp-card">
              <div className="exp-title">
                3.5 Clean Trainset & Create Pseudo Labels
              </div>
              <div className="exp-tag">Use model predictions to fix labels</div>
              <ul>
                <li>
                  For negative series in the train set with{" "}
                  <strong>“Aneurysm Present” &gt; 0.9</strong>, change the label
                  to positive and use Exp0 to create a localizer label.
                </li>
                <li>
                  For positive series in the external dataset, keep only series
                  with <strong>“Aneurysm Present” &gt; 0.5</strong> and use
                  Exp0 to create localization labels.
                </li>
              </ul>
              <div className="code-block">
{`cd src/prepare
python clean_rsna_neg.py
python create_pseudo_labeling_for_external_dataset.py`}
              </div>
            </div>

            {/* 3.6 Exp4 */}
            <div className="exp-card">
              <div className="exp-title">
                3.6 Exp4 – Classification on Cleaned + External Data
              </div>
              <div className="exp-tag">
                ViT Large & EVA Large with pseudo-labeled external data
              </div>
              <p className="exp-text">
                Two classification models are retrained on:
              </p>
              <ul>
                <li>Cleaned RSNA dataset</li>
                <li>
                  External datasets (Lausanne_TOFMRA + Royal_Brisbane_TOFMRA)
                  with pseudo labels
                </li>
              </ul>
              <div className="code-block">
{`cd exp4_cls_pseudo

# 5-fold evaluation
python train_5folds.py --cfg configs/vit_large_384.yaml
python eval_5folds.py  --cfg configs/vit_large_384.yaml

python train_5folds.py --cfg configs/eva_large_384.yaml
python eval_5folds.py  --cfg configs/eva_large_384.yaml

# Final models
python train.py --cfg configs/vit_large_384.yaml
python train.py --cfg configs/eva_large_384.yaml`}
              </div>

              <table className="metric-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>OOF</th>
                    <th>OOF + crop 0.75</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>ViT Large 384</td>
                    <td>0.8530</td>
                    <td>0.8558</td>
                  </tr>
                  <tr>
                    <td>EVA Large 384</td>
                    <td>0.8505</td>
                    <td>0.8579</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* 3.7 Exp5 */}
            <div className="exp-card">
              <div className="exp-title">
                3.7 Exp5 – Multi-task Classification + Segmentation (Cleaned +
                External)
              </div>
              <div className="exp-tag">
                MIT-B4 FPN on cleaned RSNA + external pseudo labels
              </div>
              <p className="exp-text">
                MIT-B4 FPN (image size 384) trained on the cleaned RSNA dataset
                and external datasets with pseudo labels.
              </p>
              <p className="note">
                For final predictions, again only the classification output is
                used.
              </p>
              <div className="code-block">
{`cd src/exp5_aux_pseudo
python train.py --cfg configs/mit_b4_fpn_384.yaml
python eval.py  --cfg configs/mit_b4_fpn_384.yaml`}
              </div>

              <table className="metric-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>OOF</th>
                    <th>OOF + crop 0.75</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>MIT-B4 FPN 384</td>
                    <td>0.8497</td>
                    <td>0.8629</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Final submission / ensemble */}
      <section id="final">
        <div className="section-header">
          <h2 className="section-title">4. Final Submission</h2>
          <p className="section-tagline">
            Ensemble of 6 models for robust predictions
          </p>
        </div>

        <div className="ensemble-grid">
          <div className="panel">
            <div className="panel-heading">4.1 Ensemble Recipe</div>
            <p className="panel-text">
              The final submission uses predictions from 6 models:
            </p>
            <ul>
              <li>Exp3 – MIT-B4 FPN (RSNA)</li>
              <li>Exp5 – MIT-B4 FPN (RSNA + external)</li>
              <li>Exp2 – ViT Large 384</li>
              <li>Exp4 – ViT Large 384</li>
              <li>Exp2 – EVA Large 384</li>
              <li>Exp4 – EVA Large 384</li>
            </ul>

            <div className="panel inner-panel">
              <div className="panel-heading">Final1 Ensemble</div>
              <p className="panel-text">
                The main ensemble (Final1) uses the following weights:
              </p>
              <p className="weights-text">
                <strong>Final1:</strong> 0.25 · Exp3_MIT-B4 + 0.25 · Exp5_MIT-B4
                + 0.125 · Exp2_ViT_Large + 0.125 · Exp4_ViT_Large + 0.125 ·
                Exp2_EVA_Large + 0.125 · Exp4_EVA_Large
              </p>
            </div>
          </div>

          <div className="panel">
            <div className="panel-heading">4.2 Validation Summary</div>
            <p className="panel-text">
              Local and leaderboard performance summary for Final1:
            </p>
            <table className="metric-table">
              <thead>
                <tr>
                  <th>Notebook</th>
                  <th>Local CV</th>
                  <th>Public LB</th>
                  <th>Private LB</th>
                  <th>Rank</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Final1</td>
                  <td>0.8823</td>
                  <td>0.89</td>
                  <td>0.89</td>
                  <td>-</td>
                </tr>
              </tbody>
            </table>
            <p className="note">
              For a final-year project presentation, you can focus on Local CV
              and the idea that ensembles often perform better than single
              models.
            </p>
          </div>
        </div>
      </section>

      {/* 5. Demo-Test */}
      <section id="demo">
        <div className="section-header">
          <h2 className="section-title">5. Demo – Test / Library</h2>
          <p className="section-tagline">
            Packaging the ensemble into an easier-to-use interface
          </p>
        </div>

        <div className="two-col">
          <div className="panel">
            <div className="panel-heading">5.1 Motivation</div>
            <p className="panel-text">
              Since the submission uses an ensemble of multiple models, running
              everything manually is inconvenient for other users.
            </p>
            <p className="panel-text">
              The project includes a demo notebook that packages the trained
              models into a small library-style interface. This allows:
            </p>
            <ul>
              <li>Testing new TOF-MRA series end-to-end</li>
              <li>Integrating the model into research workflows</li>
              <li>Showing a clean, single-entry-point API in presentations</li>
            </ul>
          </div>

          <div className="panel">
            <div className="panel-heading">5.2 Demo Notebook Link</div>
            <p className="panel-text">
              You can open and run the demo notebook here:
            </p>
            <p className="panel-text">
              <a
                href="https://github.com/pugazhjs9/RSNA-Intracranial-Aneurysm-Detection/blob/main/src/demo-test/test.ipynb"
                target="_blank"
                rel="noreferrer"
              >
                ./src/demo-test/test.ipynb
              </a>
            </p>
            <p className="note">
              In your video, you can show how to load a series in the notebook,
              call the inference function, and display the final “Aneurysm
              Present” probability.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div>RSNA-Intracranial-Aneurysm-Detection · Project Documentation</div>
        <div>
          <span>
            Use this page as your visual guide while recording your explanation.
          </span>
          <span className="dot-sep">
            All commands and links are taken directly from the project README.
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;
