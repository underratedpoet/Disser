import sys
import os
import traceback

import numpy as np
import torch
import librosa
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QLabel,
    QFileDialog,
    QPushButton,
    QMessageBox,
)

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QPen, QBrush

# =========================================================
# YOUR MODULES (DO NOT CHANGE)
# =========================================================

from model import AudioTCN
from feature_utils import (
    extract_features_40,
    get_log_bands,
    compute_mgd,
)

# =========================================================
# CONFIG
# =========================================================

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "tcn_audio_model.pth"

pg.setConfigOptions(antialias=True)

# =========================================================
# FEATURE MAP
# =========================================================

FEATURE_MAP = {
    0: {"type": "mgd", "band": 39},
    1: {"type": "mgd", "band": 38},
    2: {"type": "mgd", "band": 37},
    3: {"type": "mgd", "band": 33},
    4: {"type": "mgd", "band": 32},
    5: {"type": "mgd", "band": 31},
    6: {"type": "mgd", "band": 30},
    7: {"type": "mgd", "band": 28},

    12: {"type": "mag", "band": 39},
    13: {"type": "real", "band": 11},
    14: {"type": "real", "band": 0},
    15: {"type": "imag", "band": 8},

    16: {"type": "mgd", "band": 36},
    17: {"type": "mgd", "band": 35},
    18: {"type": "mgd", "band": 34},
    19: {"type": "mgd", "band": 29},
}

# =========================================================
# ANALYZER
# =========================================================

class AudioAnalyzer:

    def __init__(self, model_path):

        self.model = AudioTCN(num_features=40).to(DEVICE)

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state"])

        self.model.eval()

    def analyze(self, file_path):

        y, _ = librosa.load(
            file_path,
            sr=SR,
            mono=False
        )

        if y.ndim == 1:
            y = np.vstack([y, y])

        features = extract_features_40(
            file_path,
            target_frames=None
        )

        features_t = features.unsqueeze(0).to(DEVICE)

        features_t.requires_grad_(True)

        with torch.enable_grad():

            verdict, timeline = self.model(features_t)

            score = verdict.item()

            verdict.backward()

            saliency = (
                features_t.grad
                .detach()
                .abs()
                .squeeze(0)
                .cpu()
                .numpy()
            )

        timeline_np = (
            timeline
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        return {
            "audio": y,
            "features": features,
            "timeline": timeline_np,
            "saliency": saliency,
            "score": score,
        }

# =========================================================
# MAIN WINDOW
# =========================================================

class ForensicStudio(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Audio Forensic Studio")

        self.resize(1700, 950)

        self.analyzer = AudioAnalyzer(MODEL_PATH)

        self.current_data = None

        self.build_ui()

    # =====================================================
    # UI
    # =====================================================

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        root = QHBoxLayout(central)

        # =================================================
        # LEFT PANEL
        # =================================================

        left = QVBoxLayout()

        self.open_btn = QPushButton("OPEN AUDIO")

        self.open_btn.clicked.connect(self.open_file)

        left.addWidget(self.open_btn)

        self.score_label = QLabel("NO FILE")

        self.score_label.setAlignment(Qt.AlignCenter)

        self.score_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            padding: 10px;
        """)

        left.addWidget(self.score_label)

        self.mode_list = QListWidget()

        self.mode_list.addItems([
            "L MGD",
            "R MGD",

            "L MAG",
            "R MAG",

            "L REAL",
            "R REAL",

            "L IMAG",
            "R IMAG",

            "FEATURES"
        ])

        self.mode_list.currentRowChanged.connect(
            self.render_current
        )

        left.addWidget(self.mode_list)

        root.addLayout(left, 1)

        # =================================================
        # PLOT
        # =================================================

        self.plot = pg.PlotWidget()

        self.plot.setBackground((8, 8, 8))

        self.plot.showGrid(x=True, y=True)

        root.addWidget(self.plot, 5)

    # =====================================================
    # FILE OPEN
    # =====================================================

    def open_file(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open audio",
            "",
            "Audio (*.wav *.flac *.mp3)"
        )

        if not path:
            return

        try:

            self.load_audio(path)

            self.mode_list.setCurrentRow(0)

        except Exception as e:

            traceback.print_exc()

            QMessageBox.critical(
                self,
                "ERROR",
                str(e)
            )

    # =====================================================
    # LOAD
    # =====================================================

    def load_audio(self, path):

        result = self.analyzer.analyze(path)

        self.current_data = result

        self.file_path = path

        self.y = result["audio"]

        self.features = result["features"]

        self.timeline = result["timeline"]

        self.saliency = result["saliency"]

        self.score = result["score"]

        # =================================================
        # COMPRESSION
        # =================================================

        self.feature_frames = self.features.shape[1]

        self.model_frames = len(self.timeline)

        self.compression = (
            self.feature_frames / self.model_frames
        )

        # =================================================
        # AXES
        # =================================================

        self.time_axis = librosa.frames_to_time(
            np.arange(self.feature_frames),
            sr=SR,
            hop_length=HOP_LENGTH
        )

        self.freqs = librosa.fft_frequencies(
            sr=SR,
            n_fft=N_FFT
        )

        self.bands = get_log_bands(
            SR,
            N_FFT,
            n_bands=40
        )

        # =================================================
        # STFT
        # =================================================

        stft_l = librosa.stft(
            self.y[0],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        stft_r = librosa.stft(
            self.y[1],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        self.spec_data = {

            "L MGD": compute_mgd(
                self.y[0],
                SR,
                N_FFT
            ),

            "R MGD": compute_mgd(
                self.y[1],
                SR,
                N_FFT
            ),

            "L MAG": np.abs(stft_l),
            "R MAG": np.abs(stft_r),

            "L REAL": np.real(stft_l),
            "R REAL": np.real(stft_r),

            "L IMAG": np.imag(stft_l),
            "R IMAG": np.imag(stft_r),
        }

        # =================================================
        # LABEL
        # =================================================

        pct = self.score * 100

        color = "#ff4444" if self.score > 0.5 else "#44ff44"

        self.score_label.setText(
            f"{os.path.basename(path)}\n\n"
            f"FAKE PROBABILITY\n"
            f"{pct:.2f}%"
        )

        self.score_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {color};
            padding: 10px;
        """)

    # =====================================================
    # RENDER
    # =====================================================

    def render_current(self):

        if self.current_data is None:
            return

        mode = self.mode_list.currentItem().text()

        self.plot.clear()

        if mode == "FEATURES":

            self.render_features()

        else:

            self.render_spectrogram(mode)

    # =====================================================
    # FEATURES
    # =====================================================

    def render_features(self):

        self.plot.clear()

        self.plot.setLabel(
            "bottom",
            "Time",
            units="s"
        )

        self.plot.setLabel(
            "left",
            "Temporal Features"
        )

        self.plot.enableAutoRange()

        # =========================================
        # ONLY NON-SPECTROGRAM FEATURES
        # =========================================

        feature_indices = [
            8, 9, 10, 11,
            28, 29, 30, 31
        ]

        feature_names = [

            "L CQCC1",
            "L CQCC2",
            "L RMS",
            "L DR",

            "R CQCC1",
            "R CQCC2",
            "R RMS",
            "R DR",
        ]

        spacing = 2.0

        ticks = []

        global_saliency_mean = np.mean(
            self.saliency
        )

        for plot_idx, feat_idx in enumerate(feature_indices):

            feat = self.features[feat_idx].numpy()

            feat = (
                feat - feat.min()
            ) / (
                feat.max() - feat.min() + 1e-8
            )

            y_offset = plot_idx * spacing

            curve = feat + y_offset

            # =====================================
            # FEATURE LINE
            # =====================================

            self.plot.plot(
                self.time_axis,
                curve,
                pen=pg.mkPen(width=1)
            )

            ticks.append(
                (
                    y_offset + 0.5,
                    feature_names[plot_idx]
                )
            )

            # =====================================
            # FEATURE-SPECIFIC ANOMALIES
            # =====================================

            threshold = max(
                0.35,
                np.mean(self.timeline)
            )

            active = np.where(
                self.timeline > threshold
            )[0]

            for model_idx in active:

                feat_start = int(
                    model_idx * self.compression
                )

                feat_end = int(
                    (model_idx + 1) * self.compression
                )

                feat_start = max(
                    0,
                    min(
                        feat_start,
                        self.feature_frames - 1
                    )
                )

                feat_end = max(
                    feat_start + 1,
                    min(
                        feat_end,
                        self.feature_frames
                    )
                )

                local_sal = self.saliency[
                    feat_idx,
                    feat_start:feat_end
                ]

                if len(local_sal) == 0:
                    continue

                sal_score = np.mean(local_sal)

                if sal_score < (
                    global_saliency_mean * 1.15
                ):
                    continue

                t0 = self.time_axis[feat_start]

                t1 = self.time_axis[
                    min(
                        feat_end - 1,
                        len(self.time_axis) - 1
                    )
                ]

                rect = QRectF(
                    t0,
                    y_offset,
                    max(0.03, t1 - t0),
                    1.0
                )

                item = pg.QtWidgets.QGraphicsRectItem(
                    rect
                )

                item.setPen(
                    QPen(
                        QColor(
                            255,
                            120,
                            120,
                            90
                        )
                    )
                )

                item.setBrush(
                    QBrush(
                        QColor(
                            255,
                            0,
                            0,
                            22
                        )
                    )
                )

                self.plot.addItem(item)

        axis = self.plot.getAxis("left")

        axis.setTicks([ticks])

        self.plot.setYRange(
            -1,
            len(feature_indices) * spacing
        )

    # =====================================================
    # SPECTROGRAM
    # =====================================================

    def render_spectrogram(self, mode):

        spec = self.spec_data[mode]

        viz = librosa.amplitude_to_db(
            np.abs(spec) + 1e-7,
            ref=np.max
        )

        # [freq, time] -> [time, freq]
        viz = viz.T

        img = pg.ImageItem()

        img.setImage(viz)

        duration = self.time_axis[-1]

        # CRITICAL FIX:
        # use setRect instead of transform

        img.setRect(
            QRectF(
                0,
                0,
                duration,
                SR / 2
            )
        )

        lut = pg.colormap.get(
            "magma"
        ).getLookupTable()

        img.setLookupTable(lut)

        self.plot.addItem(img)

        self.plot.setLimits(
            xMin=0,
            xMax=duration,
            yMin=0,
            yMax=SR // 2
        )

        self.plot.setRange(
            xRange=(0, duration),
            yRange=(0, SR // 2),
            padding=0
        )

        self.plot.setLabel(
            "bottom",
            "Time",
            units="s"
        )

        self.plot.setLabel(
            "left",
            "Frequency",
            units="Hz"
        )

        self.draw_anomalies(mode)

    # =====================================================
    # DRAW ANOMALIES
    # =====================================================

    def draw_anomalies(self, mode):

        # =====================================================
        # ACTIVE MODEL FRAMES
        # =====================================================

        threshold = max(
            0.35,
            np.mean(self.timeline)
        )

        active_frames = np.where(
            self.timeline > threshold
        )[0]

        if len(active_frames) == 0:
            return

        # =====================================================
        # MODE INFO
        # =====================================================

        channel = 0 if mode.startswith("L") else 1

        spec_type = mode.split(" ")[1].lower()

        # =====================================================
        # SALIENCY THRESHOLD
        # =====================================================

        saliency_threshold = np.percentile(
            self.saliency,
            92
        )

        # =====================================================
        # PROCESS FRAMES
        # =====================================================

        for model_idx in active_frames:

            feat_start = int(
                model_idx * self.compression
            )

            feat_end = int(
                (model_idx + 1) * self.compression
            )

            feat_start = max(
                0,
                min(
                    feat_start,
                    self.feature_frames - 1
                )
            )

            feat_end = max(
                feat_start + 1,
                min(
                    feat_end,
                    self.feature_frames
                )
            )

            t0 = self.time_axis[feat_start]

            t1 = self.time_axis[
                min(
                    feat_end - 1,
                    len(self.time_axis) - 1
                )
            ]

            rect_width = max(
                0.03,
                t1 - t0
            )

            # =================================================
            # FEATURES
            # =================================================

            for feat_idx in range(40):

                ch = 0 if feat_idx < 20 else 1

                if ch != channel:
                    continue

                local_idx = feat_idx % 20

                if local_idx not in FEATURE_MAP:
                    continue

                fmap = FEATURE_MAP[local_idx]

                if fmap["type"] != spec_type:
                    continue

                # =============================================
                # LOCAL SALIENCY
                # =============================================

                local_sal = self.saliency[
                    feat_idx,
                    feat_start:feat_end
                ]

                if len(local_sal) == 0:
                    continue

                sal_score = float(
                    np.mean(local_sal)
                )

                # =============================================
                # FILTER
                # =============================================

                if sal_score < saliency_threshold:
                    continue

                # =============================================
                # FREQUENCY BAND
                # =============================================

                band_id = fmap["band"]

                band = self.bands[band_id]

                if len(band) == 0:
                    continue

                f0 = float(
                    self.freqs[band[0]]
                )
                f1 = float(
                    self.freqs[
                        min(
                            band[-1] + 1,
                            len(self.freqs) - 1
                        )
                    ]
                )

                # =============================================
                # RECTANGLE
                # =============================================

                rect_height = max(
                    40,
                    f1 - f0
                )

                rect = QRectF(
                    t0,
                    f0,
                    rect_width,
                    rect_height
                )

                print(
                    "RECT",
                    t0,
                    t1,
                    f0,
                    f1,
                    sal_score
                )
                item = pg.QtWidgets.QGraphicsRectItem(
                    rect
                )

                # =============================================
                # SUSPICIOUSNESS
                # =============================================

                timeline_score = float(
                    self.timeline[model_idx]
                )

                # normalize ONLY for visuals

                visual_score = min(
                    1.0,
                    timeline_score
                )

                # =============================================
                # COLOR
                # =============================================

                if visual_score < 0.45:

                    color = QColor(
                        0,
                        255,
                        0,
                        40
                    )

                elif visual_score < 0.7:

                    color = QColor(
                        0,
                        120,
                        120,
                        40
                    )

                else:

                    color = QColor(
                        0,
                        0,
                        255,
                        40
                    )

                # =============================================
                # PEN
                # =============================================

                pen = QPen(color)

                pen.setWidthF(
                    1.2 + visual_score * 2.8
                )

                # weak anomalies dashed

                if visual_score < 0.45:

                    pen.setStyle(Qt.DashLine)

                else:

                    pen.setStyle(Qt.DashLine)

                item.setPen(pen)

                # =============================================
                # NO FILL
                # =============================================

                #item.setBrush(QBrush(QColor(0, 0, 0, 0)))

                item.setZValue(20)

                self.plot.addItem(item)

    # =====================================================
    # TEMPORAL REGIONS
    # =====================================================

    def draw_temporal_regions(self):

        threshold = max(
            0.35,
            np.mean(self.timeline)
        )

        active = np.where(
            self.timeline > threshold
        )[0]

        if len(active) == 0:
            return

        groups = []

        start = active[0]

        for i in range(1, len(active)):

            if active[i] != active[i - 1] + 1:

                groups.append(
                    (start, active[i - 1])
                )

                start = active[i]

        groups.append(
            (start, active[-1])
        )

        for s, e in groups:

            fs = int(s * self.compression)

            fe = int((e + 1) * self.compression)

            fs = min(fs, len(self.time_axis) - 1)

            fe = min(fe, len(self.time_axis) - 1)

            t0 = self.time_axis[fs]

            t1 = self.time_axis[fe]

            region = pg.LinearRegionItem(
                [t0, t1],
                movable=False,
                brush=pg.mkBrush(
                    255,
                    0,
                    0,
                    60
                )
            )

            region.setZValue(100)

            self.plot.addItem(region)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = ForensicStudio()

    win.show()

    sys.exit(app.exec_())