import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from constants import GLOBAL_LABEL_MAP


def load_audio(file_path, target_sample_rate):
    waveform, sample_rate = torchaudio.load(file_path, normalize=True)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(
            waveform
        )

    max_audio_length = 30 * 16000
    if waveform.shape[-1] > max_audio_length:
        waveform = waveform[..., :max_audio_length]
    return waveform


class BaseAudioDataset(Dataset):
    def __init__(
        self,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.processor = processor
        self.sample_rate = sample_rate
        self.file_paths = []
        self.labels = []
        self.input_name = input_name
        self.downsample_factor = downsample_factor
        self.resampler = None
        self.actual_sample_rate = sample_rate
        try:
            if processor.sampling_rate != sample_rate:
                print("Resampling to", processor.sampling_rate, "from", sample_rate)
                self.resampler = torchaudio.transforms.Resample(
                    sample_rate, processor.sampling_rate
                )
                self.actual_sample_rate = processor.sampling_rate
        except:
            pass

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform = load_audio(file_path, self.sample_rate)
        label = self.labels[idx]

        if self.resampler:
            waveform = self.resampler(waveform)

        input_features = self.processor(
            waveform.squeeze(),
            sampling_rate=self.actual_sample_rate,
            return_tensors="pt",
        )

        start_index = 0

        end_index = waveform.shape[-1] // self.downsample_factor

        if hasattr(input_features, "input_values"):
            input_features = input_features.input_values
        elif hasattr(input_features, "input_features"):
            input_features = input_features.input_features
        else:
            raise AttributeError("Input features not found in the processor output.")

        if input_features.ndim == 3:
            input_features = input_features.squeeze(0)

        indices = torch.tensor(list(zip([start_index], [end_index])))

        label_tensor = torch.zeros(len(GLOBAL_LABEL_MAP), dtype=torch.float32)
        label_idx = GLOBAL_LABEL_MAP.get(label, GLOBAL_LABEL_MAP["other"])
        label_tensor[label_idx] = 1.0

        return input_features, label_tensor, indices

    def add_file(self, file_path, standardized_label):
        self.file_paths.append(file_path)
        self.labels.append(standardized_label)


class AudioFolderDataset(BaseAudioDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        label_map=None,
        input_name="input_values",
        downsample_factor=320,
    ):
        super().__init__(processor, sample_rate, input_name, downsample_factor)
        self.data_dir = data_dir
        self.load_data()
        self.label_map = label_map

    def load_data(self):
        for root, dirs, files in os.walk(self.data_dir):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    if self.label_map:
                        label = self.label_map.get(label, "other")
                    self.add_file(file_path, label)


class EmoDBDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "W": "anger",
            "A": "fear",
            "F": "happiness",
            "T": "sadness",
            "E": "disgust",
            "L": "sadness",
            "N": "neutral",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        emotion_letter = os.path.basename(file_path)[5]
        label = self.label_map.get(emotion_letter.upper(), "other")
        super().add_file(file_path, label)


class EMOVODataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "pau": "fear",
            "rab": "anger",
            "sor": "surprise",
            "tri": "sadness",
            "dis": "disgust",
            "gio": "happiness",
            "neu": "neutral",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        emotion = os.path.basename(file_path).split("-")[0]
        label = self.label_map.get(emotion, "other")
        super().add_file(file_path, label)


class URDUDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "Angry": "anger",
            "Neutral": "neutral",
            "Sad": "sadness",
            "Happy": "happiness",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        for label in self.label_map:
            if label.lower() in file_path.lower():
                label = self.label_map[label]
                super().add_file(file_path, label)
                return
        super().add_file(file_path, "other")


class eNTERFACEDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "anger": "anger",
            "disgust": "disgust",
            "fear": "fear",
            "happiness": "happiness",
            "sadness": "sadness",
            "surprise": "surprise",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        label = "other"
        for emotion in self.label_map.keys():
            if emotion in file_path.lower():
                label = emotion
                break
        super().add_file(file_path, label)


class MESDDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "happiness": "happiness",
            "sadness": "sadness",
            "disgust": "disgust",
            "fear": "fear",
            "surprise": "surprise",
            "anger": "anger",
            "neutral": "neutral",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        label_prefix = os.path.basename(file_path).split("_")[0].lower()
        label = next(
            (
                lbl
                for lbl, lbl_val in self.label_map.items()
                if lbl.startswith(label_prefix)
            ),
            "other",
        )
        super().add_file(file_path, label)


class MASCDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "neutral": "neutral",
            "anger": "anger",
            "elation": "happiness",
            "panic": "fear",
            "sadness": "sadness",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        label = self.label_map.get(label, "other")

        super().add_file(file_path, label)


class DEMoSDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "col": "sadness",
            "dis": "disgust",
            "gio": "happiness",
            "pau": "fear",
            "rab": "anger",
            "sor": "surprise",
            "tri": "sadness",
            "neu": "neutral",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        file_name = os.path.basename(file_path)
        emotion = file_name.split("_")[-1].split(".")[0].lower()[:3]
        label = self.label_map.get(emotion, "other")
        super().add_file(file_path, label)


class CASIADataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "happy": "happiness",
            "sad": "sadness",
            "angry": "anger",
            "fear": "fear",
            "surprise": "surprise",
            "neutral": "neutral",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )


class AESDDDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "a": "anger",
            "d": "disgust",
            "f": "fear",
            "h": "happiness",
            "s": "sadness",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        emotion_letter = os.path.basename(file_path)[0].lower()
        label = self.label_map.get(emotion_letter, "other")
        super().add_file(file_path, label)


class BAUMDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "Happiness": "happiness",
            "Sadness": "sadness",
            "Disgust": "disgust",
            "Fear": "fear",
            "Surprise": "surprise",
            "Anger": "anger",
            "Neutral": "neutral",
        }
        super().__init__(
            os.path.join(data_dir, "audio_folders"),
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )


class EEKKDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "neutral": "neutral",
            "anger": "anger",
            "joy": "happiness",
            "sadness": "sadness",
        }
        super().__init__(
            os.path.join(data_dir, "audio_folders"),
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )


class ThorstenDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "neutral": "neutral",
            "angry": "anger",
            "amused": "happiness",
            "disgusted": "disgust",
            "whisper": "other",
            "surprised": "surprise",
            "sleepy": "other",
            "drunk": "other",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )


class RESDDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "neutral": "neutral",
            "anger": "anger",
            "happiness": "happiness",
            "disgust": "disgust",
            "fear": "fear",
            "sadness": "sadness",
            "enthusiasm": "other",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )


class MELDDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "neutral": "neutral",
            "anger": "anger",
            "joy": "happiness",
            "disgust": "disgust",
            "fear": "fear",
            "sadness": "sadness",
            "surprise": "surprise",
        }
        super().__init__(
            os.path.join(data_dir, "emotion_folders"),
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )


class MEADDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "neutral": "neutral",
            "angry": "anger",
            "happy": "happiness",
            "disgusted": "disgust",
            "fear": "fear",
            "contempt": "disgust",
            "sad": "sadness",
            "surprised": "surprise",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        label = self.label_map.get(label, "other")
        super().add_file(file_path, label)


class CaFEDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "Neutre": "neutral",
            "Colère": "anger",
            "Joie": "happiness",
            "Dégoût": "disgust",
            "Peur": "fear",
            "Tristesse": "sadness",
            "Surprise": "surprise",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        if label == "CaFE_48k":
            label = os.path.basename(os.path.dirname(file_path))
        label = self.label_map.get(label, "other")
        super().add_file(file_path, label)


class ExpressoDataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "angry": "anger",
            "animal": "other",
            "animal_directed": "other",
            "awe": "surprise",
            "bored": "neutral",
            "calm": "neutral",
            "child": "other",
            "child_directed": "other",
            "confused": "other",
            "default": "neutral",
            "desire": "other",
            "disgusted": "disgust",
            "enunciated": "other",
            "fast": "other",
            "fearful": "fear",
            "happy": "happiness",
            "laughing": "happiness",
            "narration": "other",
            "non_verbal": "other",
            "projected": "other",
            "sad": "sadness",
            "sarcastic": "other",
            "singing": "other",
            "sleepy": "other",
            "sympathetic": "other",
            "whisper": "other",
        }

        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        for label in self.label_map:
            file_name = os.path.basename(file_path)
            if label.lower() in file_name.lower():
                label = self.label_map[label]
                super().add_file(file_path, label)
                return
        super().add_file(file_path, "other")


class ShEMODataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "H": "happiness",
            "S": "sadness",
            "W": "surprise",
            "A": "anger",
            "N": "neutral",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        label = os.path.basename(file_path)[3].upper()
        label = self.label_map.get(label, "other")
        super().add_file(file_path, label)


class SUBESCODataset(AudioFolderDataset):
    def __init__(
        self,
        data_dir,
        processor,
        sample_rate=16000,
        input_name="input_values",
        downsample_factor=320,
    ):
        self.label_map = {
            "NEUTRAL": "neutral",
            "ANGRY": "anger",
            "HAPPY": "happiness",
            "DISGUST": "disgust",
            "FEAR": "fear",
            "SAD": "sadness",
            "SURPRISE": "surprise",
        }
        super().__init__(
            data_dir,
            processor,
            sample_rate,
            self.label_map,
            input_name,
            downsample_factor,
        )

    def add_file(self, file_path, _):
        label = os.path.basename(file_path).split("_")[-2]
        label = self.label_map.get(label, "other")
        super().add_file(file_path, label)
