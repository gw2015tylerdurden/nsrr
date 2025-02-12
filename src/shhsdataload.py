import os
import glob
import csv
from xml.etree import ElementTree
from src.resample import Resample
from src.normalize import Normalizer
import pyedflib
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import ListConfig


class ShhsDataLoader:
    def __init__(self, annotation_labels, base_path='./shhs/polysomnography/', datasets=['shhs1', 'shhs2'],  upsampling='linear', norm='standard', annot='nsrr', output_csv=False, verbose=True):
        self.annotation_labels = annotation_labels
        self.verbose = verbose
        self.edf_files = []
        self.xml_files = []
        self.resample = Resample.get_instance(upsampling)
        self.normalizer = Normalizer.get_instance(norm)
        self.duration = 30.0

        get_all_files = lambda path, pattern: glob.glob(os.path.join(path, '**', pattern), recursive=True)
        for dataset in datasets:
            edf_path = os.path.expanduser(os.path.join(base_path, "edfs", dataset))
            xml_path = os.path.expanduser(os.path.join(base_path, "annotations-events-" + annot, dataset))
            edf_files = get_all_files(edf_path, '*.edf')
            xml_files = get_all_files(xml_path, '*.xml')

            if self.verbose:
                print_file_info = lambda files, file_type: print(
                 f"[INFO] Total {file_type} files: {len(files)}\n[INFO] Total size: {sum(os.path.getsize(file) for file in files) / (1024**3):.2f} GB")
                print(f"[INFO] dataset for {dataset}")
                print_file_info(edf_files, "EDF")
                print_file_info(xml_files, "XML")

            self.edf_files.extend(edf_files)
            self.xml_files.extend(xml_files)

        if not self.__check_all_files_match():
            exit()

        if output_csv:
            self.generate_annot_counts_csv()

    def __check_all_files_match(self):
        edf_basenames = {os.path.splitext(os.path.basename(p))[0] for p in self.edf_files}
        # remove '-nsrr.xml' form xml filename to compare edf_basemase
        xml_basenames = {os.path.basename(p).rsplit('-', 1)[0] for p in self.xml_files}


        # 集合（set）の差分を取ることで、一方の集合に存在して、もう一方に存在しない要素を取得するための処理
        unmatched_edf = edf_basenames - xml_basenames
        unmatched_xml = xml_basenames - edf_basenames

        if not unmatched_edf and not unmatched_xml:
            if self.verbose:
                print("[INFO] EDF files are matched with XML files")
            return True
        else:
            if unmatched_edf:
                print("[ERROR] EDF files are NOT matched with XML files:", unmatched_edf)
            if unmatched_xml:
                print("[ERROR] XML files are NOT matched with EDF files:", unmatched_xml)
            return False

    def generate_annot_counts_csv(self, output_file='shhs-annot-counts.csv'):
        total_label_counts = {label: 0 for label in self.annotation_labels}

        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            headers = ["xml_file"] + self.annotation_labels
            csvwriter.writerow(headers)

            for i, xml_file in enumerate(self.xml_files):
                tree = ElementTree.parse(xml_file)
                label_counts = {label: 0 for label in self.annotation_labels}

                # read annotation label from xml
                for event in tree.findall('.//ScoredEvent'):
                    concept = event.find('EventConcept').text
                    concept = concept.split('|')[0].strip()
                    duration = event.find('Duration')

                    increment_count = 1
                    if duration is not None and concept in self.annotation_labels:
                        duration_value = float(duration.text)
                        if duration_value % 30.0 == 0:
                            increment_count = int(duration_value / 30.0)
                        else:
                            print(f"[WARN] duration % 30 != 0: duration{duration}:xml_file{xml_file}:label{concept}")

                    if concept in label_counts:
                        label_counts[concept] += increment_count
                        total_label_counts[concept] += increment_count

                # read channel label from edf
                with pyedflib.EdfReader(self.edf_files[i]) as f:
                    edf_labels = f.getSignalLabels()

                row = [xml_file] + [label_counts[label] for label in self.annotation_labels] + edf_labels
                csvwriter.writerow(row)

        if self.verbose:
            count_results = [total_label_counts[label] for label in self.annotation_labels]
            print("------------------------------------")
            print("[INFO] Total counts:")
            for label, count in zip(self.annotation_labels, count_results):
                print(f"{label}: {count}")
            print("------------------------------------")

        print(f"[INFO] create {output_file}")

    def create_target_fs_dataset_h5(self, channel_labels, fs_channels, target_fs=None, creation_dataset_path='./', debug_plots_interval=None, save_file_interval=1000):
        total_count = 0
        h5_file_index = 0
        hf = None
        self.fs_channels = fs_channels
        self.channel_labels = channel_labels
        self.annotation_counts = {label: 0 for label in self.annotation_labels}
        self.debug_plotter = PreprocessResultPlotter(debug_plots_interval, channel_labels)
        os.makedirs(creation_dataset_path, exist_ok=True)
        # for save variable length signals in h5
        # pyedflib.readSignal returns float64, but original EDF has a 16 bit data format
        if target_fs is None:
            dtype_variable_float = h5py.special_dtype(vlen=np.dtype('float16'))
        else:
            dtype_variable_float = np.dtype('float16')


        for idx, (edf_file, xml_file) in enumerate(tqdm(zip(self.edf_files, self.xml_files), total=len(self.edf_files), desc="Processing files")):
            if idx % save_file_interval == 0:
                if 'hf' in locals() and hf is not None:
                    # write data information when closing h5 file
                    hf.create_dataset(f"fs_channels", data=self.fs_channels, dtype=np.float32)
                    hf.create_dataset(f"channels", data=str(self.channel_labels[0]) if isinstance(self.channel_labels, list) else str(self.channel_labels))
                    hf.create_dataset(f"target_fs", data=str(target_fs))
                    hf.create_dataset(f"annotation_labels", data=str(self.annotation_labels))
                    hf.close()
                filename = f'dataset_{h5_file_index}.h5'
                hf = h5py.File(creation_dataset_path + filename, 'w')
                h5_file_index += 1

            has_edf_all_target_channnels, fs_channels, signals = self.__load_edf_file_target_channel(edf_file)
            if has_edf_all_target_channnels is False:
                continue

            signal_list, label_list = self.__preprocessing(xml_file, fs_channels, signals, target_fs, total_count)

            patient_no = f"{os.path.splitext(os.path.basename(edf_file))[0]}"
            for (signal, label) in zip(signal_list, label_list):
                dataset_name = f"shhs{total_count}"
                hf.create_dataset(f"{patient_no}/{dataset_name}/signal", data=signal, dtype=dtype_variable_float)
                hf.create_dataset(f"{patient_no}/{dataset_name}/label", data=label, dtype=np.uint8)
                total_count += 1

        if 'hf' in locals() and hf is not None:
            hf.close()

        if self.verbose:
            print("------------------------------------")
            print(f"[INFO] Total created data counts: {total_count}")
            for label, count in self.annotation_counts.items():
                print(f"{label}: {count}")
            print("------------------------------------")



    def __preprocessing(self, xml_file, fs_channels, signals, target_fs, total_count):
        signal_list = []
        label_list = []
        debug_plots_interval = total_count

        tree = ElementTree.parse(xml_file)
        for event in tree.findall('.//ScoredEvent'):
            annotation = event.find('EventConcept').text.split('|')[0].strip()
            start_time = float(event.find('Start').text)
            durations = float(event.find('Duration').text)

            if annotation in self.annotation_labels:

                measurement_time = len(signals[0]) / fs_channels[0]
                if start_time + durations > measurement_time:
                    # situation that durations is longer than signal
                    num_iterations = int((measurement_time - start_time) / self.duration)
                else:
                    num_iterations = int(durations / self.duration)

                for count in range(num_iterations):
                    extracted_signals = self.__extract_data(fs_channels, signals, start_time, count)
                    normalized_signals = [self.normalizer.normalize(signal) for signal in extracted_signals]
                    if target_fs is None:
                        #after_fs_signals = self.__padding_signals(extracted_signals)
                        after_fs_signals = normalized_signals
                    else:
                        after_fs_signals = self.__upsample_data(fs_channels, normalized_signals, self.duration, target_fs)

                    label_idx = self.annotation_labels.index(annotation)
                    label_list.append(label_idx)
                    signal_list.append(normalized_signals)

                    # to confirm result
                    self.annotation_counts[annotation] += 1  # for verbose
                    self.debug_plotter.plot_and_save(debug_plots_interval, fs_channels, target_fs, self.duration, annotation, extracted_signals, after_fs_signals, normalized_signals)
                    debug_plots_interval += 1

        return signal_list, label_list


    def __padding_signals(self, signals):
        padding_signals = []
        max_length = max(len(s) for s in signals)

        for signal in signals:
            padding_length = max_length - len(signal)
            # padding NaN
            padding_signal = np.concatenate([signal, np.full(padding_length, np.nan)])
            padding_signals.append(padding_signal)
        return padding_signals

    def __is_all_channel_included(self, edf_channels):
        for label in self.channel_labels:
            if isinstance(label, ListConfig) or isinstance(label, list):
                if not any(item in edf_channels for item in label):
                    return False
            else:
                if label not in edf_channels:
                    return False
        return True

    def __load_edf_file_target_channel(self, edf_file):
        fs = []
        signals = []

        with pyedflib.EdfReader(edf_file) as f:
            # load all channel labels
            edf_channels = f.getSignalLabels()
            if not self.__is_all_channel_included(edf_channels):
                # lack target channel in xml
                return False, None, None

            for target_label in self.channel_labels:
                if isinstance(target_label, ListConfig) or isinstance(target_label, list):
                    for i in target_label:
                        try:
                            channel_idx = edf_channels.index(i)
                            break
                        except:
                            continue
                else:
                    channel_idx = edf_channels.index(target_label)
                fs.append(f.samplefrequency(channel_idx))
                signals.append(f.readSignal(channel_idx))

            if self.fs_channels != fs:
                if self.verbose:
                    print(f"[WARN] sampling freq is not correct({self.fs_channels}): {edf_file}:{fs}")
                return False, None, None

        return True, fs, signals


    def __extract_data(self, fs_channels, signals, start_time, count):
        extracted_data = []

        for idx, fs in enumerate(fs_channels):
            start_idx = int(fs * (start_time + count * self.duration))
            end_idx = start_idx + int(fs * (self.duration))

            if end_idx <= len(signals[idx]):
                # extract data of the specified channel for the duration range
                data = signals[idx][start_idx:end_idx]
                extracted_data.append(data)
            else:
                print('duration is larger than the data length')
                # do not append data anymore
                break

        return extracted_data


    def __upsample_data(self, fs_channels, extracted_data, duration, target_fs):
        resampled_data = []
        resampled_time = np.linspace(0, duration, int(target_fs * duration))

        for data, fs in zip(extracted_data, fs_channels):
            time = np.linspace(0, duration, len(data))
            resampled_data.append(self.resample.upsampling(data, time, resampled_time, fs, target_fs))

        return resampled_data


class PreprocessResultPlotter:
    def __init__(self, plots_interval, channel_labels):
        if plots_interval is None or not isinstance(plots_interval, int):
            self.is_plot = False
        else:
            self.is_plot = True
            self.plots_interval = plots_interval
            self.channel_labels = channel_labels

    def plot_and_save(self, total_count, fs_channels, target_fs, duration, annotation, extracted_signals, interpolated_signals, normalized_signals, output_format='pdf', dirname='debug_plot'):
        if not self.is_plot or total_count % self.plots_interval != 0:
            return

        #self.time = [np.arange(len(data)) / fs for data, fs in zip(extracted_signals, fs_channels)]
        self.time = [np.linspace(0, duration, len(data)) for data in extracted_signals]
        if target_fs is not None:
            self.resampled_time = np.linspace(0, duration, int(target_fs * duration))

        # Check if directory exists, if not create it
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        n_channels = len(fs_channels)
        fig, axs = plt.subplots(n_channels, 2, figsize=(50, 5 * n_channels))
        plt.rc('font', size=25)
        plt.rc('axes', titlesize=25)
        plt.rc('axes', labelsize=25)
        plt.rc('legend', fontsize=20)
        plt.rc('figure', titlesize=25)
        for ax in axs.ravel():
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(20)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(20)

        for i in range(n_channels):
            # Original Data and Interpolated Data plots
            axs[i, 0].plot(self.time[i], extracted_signals[i], label='Original Data')
            if target_fs is None:
                time = self.time[i]
            else:
                time = self.resampled_time

            axs[i, 0].plot(time, interpolated_signals[i], label='Interpolated Data', alpha=0.7)
            axs[i, 0].set_title(f"Channel: {self.channel_labels[i]} (fs: {fs_channels[i]} Hz)")
            axs[i, 0].legend(loc='upper right')

            # Normalized Signal plots
            axs[i, 1].plot(time, normalized_signals[i], label='Normalized Signal')
            if target_fs is not None:
                axs[i, 1].set_title(f"Channel: {self.channel_labels[i]} Normalized (fs: {target_fs} Hz)")
            axs[i, 1].legend(loc='upper right')

        fig.suptitle(f"Annotation: {annotation}, Total count: {total_count}")
        plt.tight_layout()

        # Save the plot inside the specified directory
        save_path = os.path.join(dirname, f"debug_plot_{total_count}.{output_format}")
        plt.savefig(save_path)
        plt.close()
