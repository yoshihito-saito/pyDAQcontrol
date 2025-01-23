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
    """
    NI-DAQmxで連続サンプリング。
    timesReady: shape=(n_samples,) (各サンプル時刻)
    dataReady: shape=(n_channels, n_samples)
    """
    timesReady = QtCore.pyqtSignal(np.ndarray)
    dataReady = QtCore.pyqtSignal(np.ndarray)

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
                # 1回あたり 100サンプル取得
                data = self.task.read(number_of_samples_per_channel=100, timeout=5.0)
                data = np.array(data, dtype=np.float64)

                # チャネル1つの場合は (1, n_samples) に reshape
                if data.ndim == 1:
                    data = data.reshape((1, -1))

                # サンプル数
                actual_samples = data.shape[1]

                # 今回のブロックの開始~終了時刻 (相対時間)
                t_now = time.time() - self.start_time
                t_start_block = t_now
                t_end_block = t_now + (actual_samples - 1)/self.sample_rate

                times = np.linspace(t_start_block, t_end_block, actual_samples)

                # シグナル発行でメインスレッドへ
                self.timesReady.emit(times)
                self.dataReady.emit(data)

                # CPU負荷軽減
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
        self.setWindowTitle("DAQ Logging - 20sec Display & Full Streaming Save")

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

        # プロット関係(リングバッファ表示用)
        self.plot_widgets = []
        self.curves = []
        self.plot_times = []
        self.plot_data = []

        self.selected_channels = []
        self.daq_thread = None

        # 逐次書き込み用の状態
        self.file_path = None
        self.file_format = None
        self.file_opened = False
        self.csv_header_written = False
        self.h5file = None  # h5py.Fileオブジェクト

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

    # -------------- Start/Stop ------------------
    def start_recording(self):
        if self.daq_thread and self.daq_thread.is_running:
            return

        # プロットクリア
        self.clear_plots()

        selected_items = self.channel_list_widget.selectedItems()
        channels = [item.text() for item in selected_items]
        if not channels:
            QtWidgets.QMessageBox.warning(self, "Warning", "No channels selected.")
            return

        sample_rate = float(self.sample_rate_lineedit.text())

        # チャンネル名保持
        self.selected_channels = channels[:]

        # PlotWidget生成(リングバッファ表示用)
        for ch_idx, ch_name in enumerate(channels):
            plot_w = pg.PlotWidget()
            plot_w.setMinimumHeight(200)
            plot_w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

            plot_w.setYRange(-1.2, 1.2)
            plot_w.setLabel("left", "Voltage")
            plot_w.setLabel("bottom", "Time (s)")
            plot_w.setTitle(ch_name)

            curve = plot_w.plot(pen="cyan")
            if ch_idx > 0:
                plot_w.setXLink(self.plot_widgets[0])
                plot_w.setYLink(self.plot_widgets[0])

            self.plot_layout.addWidget(plot_w)
            self.plot_widgets.append(plot_w)
            self.curves.append(curve)
            self.plot_times.append(np.array([], dtype=np.float64))
            self.plot_data.append(np.array([], dtype=np.float64))

        # ファイルをオープン(逐次書き込み)
        if not self.open_file_for_writing():
            return

        # DAQ スレッド開始
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

        # 最後にファイルを閉じる
        self.close_file()

        QtWidgets.QMessageBox.information(self, "Info", "DAQ stopped and all data saved.")

    def clear_plots(self):
        while self.plot_widgets:
            w = self.plot_widgets.pop()
            self.plot_layout.removeWidget(w)
            w.deleteLater()
        self.curves.clear()
        self.plot_times.clear()
        self.plot_data.clear()

    # -------------- File Opening/Closing ------------------
    def open_file_for_writing(self):
        """
        新規or追記のファイルを開いて、以後 on_data_ready() ごとに書き込む。
        """
        save_dir = self.save_dir_lineedit.text()
        file_name = self.filename_lineedit.text().strip()
        if not file_name:
            file_name = "mydata"

        if not save_dir or not os.path.isdir(save_dir):
            QtWidgets.QMessageBox.warning(self, "Warning", "Invalid save directory.")
            return False

        self.file_format = self.format_combobox.currentText()  # "CSV" or "H5"
        file_path = os.path.join(save_dir, file_name)
        if self.file_format == "CSV":
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            # CSV: ヘッダの有無だけ管理
            self.csv_header_written = False

            # appendモードで開く場合、ここでは不要(都度 open("a") でも可)
            # ただし最初にファイルが存在したらどうするか等、要設計。
            self.file_path = file_path
            self.file_opened = True  # フラグだけ

        else:
            if not file_path.lower().endswith(".h5"):
                file_path += ".h5"
            self.file_path = file_path

            # 新規作成("w") or 追記("r+")
            # ここでは毎回新規作成する例
            self.h5file = h5py.File(file_path, "w")
            # timesとdataデータセットを可変長で用意
            self.h5file.create_dataset("times", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
            self.h5file.create_dataset(
                "data",
                shape=(len(self.selected_channels), 0),
                maxshape=(len(self.selected_channels), None),
                dtype=np.float64,
                chunks=(len(self.selected_channels), 1024)
            )
            # channel_names
            dt = h5py.special_dtype(vlen=str)
            ds = self.h5file.create_dataset("channel_names", (len(self.selected_channels),), dtype=dt)
            for i, ch_name in enumerate(self.selected_channels):
                ds[i] = ch_name
            self.file_opened = True

        return True

    def close_file(self):
        """ Stop時にファイルを閉じる """
        if self.file_opened:
            if self.file_format == "H5" and self.h5file is not None:
                self.h5file.close()
                self.h5file = None
            self.file_opened = False

    # -------------- DAQ -> GUI & File --------------
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

        # -------------------
        # 1) リングバッファ: 直近20秒だけ表示
        # -------------------
        max_hold_sec = 20.0
        t_latest = times_chunk[-1]
        t_threshold = t_latest - max_hold_sec

        for ch_i in range(n_channels):
            self.plot_times[ch_i] = np.concatenate([self.plot_times[ch_i], times_chunk])
            self.plot_data[ch_i] = np.concatenate([self.plot_data[ch_i], data_chunk[ch_i]])
            valid_mask = (self.plot_times[ch_i] >= t_threshold)
            self.plot_times[ch_i] = self.plot_times[ch_i][valid_mask]
            self.plot_data[ch_i] = self.plot_data[ch_i][valid_mask]

        # -------------------
        # 2) ファイルへのストリーミング保存
        # -------------------
        if self.file_opened:
            if self.file_format == "CSV":
                self.append_csv(times_chunk, data_chunk)
            else:
                self.append_hdf5(times_chunk, data_chunk)

    # -------------- CSV Append --------------
    def append_csv(self, times_chunk, data_chunk):
        """
        CSVに追記(append)する形で書き込む。
        まだヘッダを書いていない場合は書く。
        """
        file_exists = os.path.isfile(self.file_path)
        mode = "a"  # 追記モード
        with open(self.file_path, mode, newline="") as f:
            writer = csv.writer(f)
            if (not self.csv_header_written) and (not file_exists):
                # ヘッダ
                header = ["time"] + list(self.selected_channels)
                writer.writerow(header)
                self.csv_header_written = True

            n_channels, n_samps = data_chunk.shape
            for i in range(n_samps):
                row = [times_chunk[i]]
                for ch_i in range(n_channels):
                    row.append(data_chunk[ch_i, i])
                writer.writerow(row)

    # -------------- HDF5 Append (Resize) --------------
    def append_hdf5(self, times_chunk, data_chunk):
        if not self.h5file:
            return
        dset_times = self.h5file["times"]
        dset_data  = self.h5file["data"]

        old_size_t = dset_times.shape[0]
        new_size_t = old_size_t + times_chunk.size
        dset_times.resize((new_size_t,))
        dset_times[old_size_t : new_size_t] = times_chunk

        old_size_d = dset_data.shape[1]
        new_size_d = old_size_d + data_chunk.shape[1]
        dset_data.resize((len(self.selected_channels), new_size_d))
        dset_data[:, old_size_d : new_size_d] = data_chunk

        self.h5file.flush()

    # -------------- Plot Update --------------
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
