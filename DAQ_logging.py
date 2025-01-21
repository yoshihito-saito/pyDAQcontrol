import sys
import time
import numpy as np
import csv
import os

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.system import System

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class DAQWorker(QtCore.QThread):
    timesReady = QtCore.pyqtSignal(np.ndarray)  # shape=(n_samples,)
    dataReady = QtCore.pyqtSignal(np.ndarray)   # shape=(n_channels, n_samples)

    def __init__(self, channels, sample_rate=1000, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.sample_rate = sample_rate
        self.is_running = False

    def run(self):
        self.start_time = time.time()

        self.task = nidaqmx.Task()
        for ch in self.channels:
            self.task.ai_channels.add_ai_voltage_chan(ch)

        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS
        )

        self.task.start()
        self.is_running = True

        try:
            while self.is_running:
                data = self.task.read(number_of_samples_per_channel=100, timeout=5.0)
                data = np.array(data, dtype=np.float64)
                if data.ndim == 1:
                    data = data.reshape((1, -1))

                actual_samples = data.shape[1]
                t_now = time.time() - self.start_time
                t_start_block = t_now
                t_end_block = t_now + (actual_samples - 1)/self.sample_rate

                times = np.linspace(t_start_block, t_end_block, actual_samples)

                self.timesReady.emit(times)
                self.dataReady.emit(data)

                time.sleep(0.01)
        finally:
            self.task.stop()
            self.task.close()

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait(1000)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DAQ Logging")

        self.setFixedSize(1000, 1000)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        # -------------------------------------------------
        # 上部: チャンネル選択 & 設定
        # -------------------------------------------------
        config_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(config_layout)

        config_layout.addWidget(QtWidgets.QLabel("Select Channels:"))
        self.channel_list_widget = QtWidgets.QListWidget()
        self.channel_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channel_list_widget.setFixedHeight(120)
        config_layout.addWidget(self.channel_list_widget)

        self.load_device_channels()

        config_layout.addWidget(QtWidgets.QLabel("Sample Rate (Hz):"))
        self.sample_rate_lineedit = QtWidgets.QLineEdit("1000")
        self.sample_rate_lineedit.setFixedWidth(80)
        config_layout.addWidget(self.sample_rate_lineedit)

        config_layout.addWidget(QtWidgets.QLabel("Time Scale (sec):"))
        self.time_scale_lineedit = QtWidgets.QLineEdit("5")
        self.time_scale_lineedit.setFixedWidth(50)
        config_layout.addWidget(self.time_scale_lineedit)

        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        config_layout.addWidget(self.start_button)
        config_layout.addWidget(self.stop_button)

        # -------------------------------------------------
        # 中段: 保存先設定
        # -------------------------------------------------
        save_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(save_layout)

        save_layout.addWidget(QtWidgets.QLabel("Save Dir:"))
        self.save_dir_lineedit = QtWidgets.QLineEdit("")
        self.save_dir_lineedit.setFixedWidth(300)
        save_layout.addWidget(self.save_dir_lineedit)

        self.browse_button = QtWidgets.QPushButton("Browse...")
        save_layout.addWidget(self.browse_button)

        save_layout.addWidget(QtWidgets.QLabel("File Name:"))
        self.filename_lineedit = QtWidgets.QLineEdit("mydata")
        self.filename_lineedit.setFixedWidth(150)
        save_layout.addWidget(self.filename_lineedit)

        save_layout.addWidget(QtWidgets.QLabel("Format:"))
        self.format_combobox = QtWidgets.QComboBox()
        self.format_combobox.addItems(["CSV", "H5"])
        save_layout.addWidget(self.format_combobox)

        # -------------------------------------------------
        # Plotを置くスクロールエリア
        # -------------------------------------------------
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        self.plot_container = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.plot_container)
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_container.setLayout(self.plot_layout)

        # プロット関係
        self.plot_widgets = []
        self.curves = []
        self.plot_times = []
        self.plot_data = []

        # ★ 追加: ユーザが選択したチャンネル名を保持
        self.selected_channels = []

        # スレッド
        self.daq_thread = None

        # イベント
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.browse_button.clicked.connect(self.browse_save_dir)

        # タイマー
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()

    def load_device_channels(self):
        system = System.local()
        for dev in system.devices:
            for ai_chan in dev.ai_physical_chans:
                chan_name = f"{ai_chan.name}"
                item = QtWidgets.QListWidgetItem(chan_name)
                self.channel_list_widget.addItem(item)

    def browse_save_dir(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if dir_path:
            self.save_dir_lineedit.setText(dir_path)

    def start_recording(self):
        if self.daq_thread and self.daq_thread.is_running:
            return

        # 前のプロットクリア
        self.clear_plots()

        selected_items = self.channel_list_widget.selectedItems()
        channels = [item.text() for item in selected_items]
        if not channels:
            QtWidgets.QMessageBox.warning(self, "Warning", "No channels selected.")
            return

        sample_rate = float(self.sample_rate_lineedit.text())

        # ★ 選択したチャネル名を保持
        self.selected_channels = channels[:]  # コピー

        # PlotWidget生成
        for ch_idx, ch_name in enumerate(channels):
            plot_w = pg.PlotWidget()
            plot_w.setMinimumHeight(200)
            plot_w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

            plot_w.setYRange(-1.2, 1.2)
            plot_w.setLabel("left", "Voltage")
            plot_w.setLabel("bottom", "Time (s)")
            plot_w.setTitle(ch_name)

            curve = plot_w.plot(pen="cyan")
            if ch_idx == 0:
                pass
            else:
                plot_w.setXLink(self.plot_widgets[0])
                plot_w.setYLink(self.plot_widgets[0])

            self.plot_layout.addWidget(plot_w)
            self.plot_widgets.append(plot_w)
            self.curves.append(curve)

            self.plot_times.append(np.array([], dtype=np.float64))
            self.plot_data.append(np.array([], dtype=np.float64))

        self.daq_thread = DAQWorker(channels=channels, sample_rate=sample_rate)
        self.daq_thread.timesReady.connect(self.on_times_ready)
        self.daq_thread.dataReady.connect(self.on_data_ready)
        self.daq_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recording(self):
        if self.daq_thread:
            self.daq_thread.stop()
            self.daq_thread = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # データ保存
        self.save_data()

    def clear_plots(self):
        while self.plot_widgets:
            w = self.plot_widgets.pop()
            self.plot_layout.removeWidget(w)
            w.deleteLater()
        self.curves.clear()
        self.plot_times.clear()
        self.plot_data.clear()

    def on_times_ready(self, times):
        self._current_times = times

    def on_data_ready(self, data_chunk):
        if not hasattr(self, "_current_times"):
            return
        times_chunk = self._current_times
        del self._current_times

        n_channels, n_samples = data_chunk.shape
        if len(self.plot_times) != n_channels:
            return

        max_hold_sec = 20.0
        t_latest = times_chunk[-1]
        t_threshold = t_latest - max_hold_sec

        for ch_i in range(n_channels):
            self.plot_times[ch_i] = np.concatenate([self.plot_times[ch_i], times_chunk])
            self.plot_data[ch_i] = np.concatenate([self.plot_data[ch_i], data_chunk[ch_i]])
            valid_mask = (self.plot_times[ch_i] >= t_threshold)
            self.plot_times[ch_i] = self.plot_times[ch_i][valid_mask]
            self.plot_data[ch_i] = self.plot_data[ch_i][valid_mask]

    def update_plots(self):
        if not self.curves:
            return
        try:
            time_scale = float(self.time_scale_lineedit.text())
        except ValueError:
            time_scale = 5.0

        t_now = 0.0
        if self.daq_thread:
            t_now = time.time() - self.daq_thread.start_time
        else:
            t_now = time.time()

        t_min = t_now - time_scale
        if t_min < 0.0:
            t_min = 0.0

        for ch_i, curve in enumerate(self.curves):
            t_array = self.plot_times[ch_i]
            y_array = self.plot_data[ch_i]
            valid_mask = (t_array >= t_min) & (t_array <= t_now)
            t_valid = t_array[valid_mask]
            y_valid = y_array[valid_mask]

            if t_valid.size == 0:
                curve.setData([], [])
                continue

            x_data = t_valid - t_min
            curve.setData(x_data, y_valid)

            if ch_i < len(self.plot_widgets):
                self.plot_widgets[ch_i].setXRange(0, time_scale, padding=0)

    # ------------------ データ保存 -------------------
    def save_data(self):
        save_dir = self.save_dir_lineedit.text()
        file_name = self.filename_lineedit.text().strip()
        if not file_name:
            file_name = "mydata"

        save_format = self.format_combobox.currentText()  # "CSV" or "H5"

        if not save_dir or not os.path.isdir(save_dir):
            QtWidgets.QMessageBox.warning(self, "Warning", "Invalid save directory.")
            return

        n_channels = len(self.plot_times)
        if n_channels == 0:
            QtWidgets.QMessageBox.information(self, "Info", "No data to save.")
            return

        lengths = [len(self.plot_times[ch]) for ch in range(n_channels)]
        max_len = max(lengths)

        # 代表として一番長い times
        rep_i = np.argmax(lengths)
        rep_times = self.plot_times[rep_i]
        out_times = rep_times.copy()
        out_data = np.full((n_channels, max_len), np.nan, dtype=np.float64)

        for ch_i in range(n_channels):
            length_i = len(self.plot_times[ch_i])
            if length_i == max_len:
                out_data[ch_i, :] = self.plot_data[ch_i]
            else:
                # 小さいほうに合わせる
                out_data[ch_i, :length_i] = self.plot_data[ch_i]

        file_path = os.path.join(save_dir, file_name)
        if save_format == "CSV":
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            self.save_csv(file_path, out_times, out_data)
        else:
            if not file_path.lower().endswith(".h5"):
                file_path += ".h5"
            if not HAS_H5PY:
                QtWidgets.QMessageBox.warning(self, "Warning", "h5py is not installed.")
                return
            self.save_hdf5(file_path, out_times, out_data)

        QtWidgets.QMessageBox.information(self, "Info", f"Data saved to {file_path}")

    def save_csv(self, file_path, times, data):
        # ヘッダにself.selected_channelsを使用
        n_channels = len(self.selected_channels)
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            # 先頭列は "time", その後に選択したチャネル名をそのまま
            header = ["time"] + list(self.selected_channels)
            writer.writerow(header)

            max_len = times.size
            for i in range(max_len):
                row = [times[i]]
                for ch_i in range(n_channels):
                    row.append(data[ch_i, i])
                writer.writerow(row)

    def save_hdf5(self, file_path, times, data):
        import h5py
        n_channels = len(self.selected_channels)
        with h5py.File(file_path, "w") as f:
            f.create_dataset("times", data=times, dtype=np.float64)
            dset = f.create_dataset("data", data=data, dtype=np.float64)
            # チャネル名の保存
            # HDF5で文字列リストを保存するには、可変長文字列dtypeを使う方法などがある
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("channel_names", (n_channels,), dt)
            for i in range(n_channels):
                f["channel_names"][i] = self.selected_channels[i]


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

