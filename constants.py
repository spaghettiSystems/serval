import torch

from whisper_flash_attention import WhisperFlashAttentionForConditionalGeneration
from transformers import AutoModel, ClapFeatureExtractor, ClapModel, HubertModel, SeamlessM4TFeatureExtractor, Wav2Vec2BertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model, WavLMModel, WhisperProcessor
DATASET_DISTRIBUTIONS = {
    'URDUDataset': torch.tensor([0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25]),
    'EmoDBDataset': torch.tensor([0.13271028037383178, 0.2672897196261682, 0.08598130841121496, 0.12897196261682242, 0.0, 0.23738317757009345, 0.0, 0.14766355140186915]),
    'EMOVODataset': torch.tensor([0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.0, 0.14285714285714285]),
    'eNTERFACEDataset': torch.tensor([0.16473317865429235, 0.16705336426914152, 0.16705336426914152, 0.16705336426914152, 0.16705336426914152, 0.16705336426914152, 0.0, 0.0]),
    'MESDDataset': torch.tensor([0.16695652173913045, 0.16695652173913045, 0.16695652173913045, 0.16695652173913045, 0.0, 0.16608695652173913, 0.0, 0.16608695652173913]),
    'MASCDataset': torch.tensor([0.1989389920424403, 0.1989389920424403, 0.0, 0.1989389920424403, 0.0, 0.1989389920424403, 0.0, 0.20424403183023873]),
    'DEMoSDataset': torch.tensor([0.143858925440858, 0.2742085180983809, 0.17304320923997113, 0.11921212746210168, 0.10312467773538207, 0.15231514901515933, 0.0, 0.03423739300814685]),
    'CASIADataset': torch.tensor([0.16666666666666666, 0.16666666666666666, 0.0, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.0, 0.16666666666666666]),
    'AESDDDataset': torch.tensor([0.19701986754966888, 0.20198675496688742, 0.20198675496688742, 0.1986754966887417, 0.0, 0.20033112582781457, 0.0, 0.0]),
    'BAUMDataset': torch.tensor([0.14234620886981403, 0.12160228898426323, 0.0815450643776824, 0.05150214592274678, 0.029327610872675252, 0.07010014306151645, 0.37124463519313305, 0.1323319027181688]),
    'EEKKDataset': torch.tensor([0.23281786941580757, 0.21391752577319587, 0.0, 0.0, 0.0, 0.26288659793814434, 0.0, 0.29037800687285226]),
    'ThorstenDataset': torch.tensor([0.12505210504376824, 0.0, 0.12505210504376824, 0.0, 0.12505210504376824, 0.12505210504376824, 0.3747394747811588, 0.12505210504376824]),
    'RESDDataset': torch.tensor([0.15616045845272206, 0.11604584527220631, 0.13252148997134672, 0.15974212034383956, 0.0, 0.15687679083094555, 0.14183381088825214, 0.13681948424068768]),
    'MELDDataset': torch.tensor([0.17293407613741876, 0.0766016713091922, 0.027855153203342618, 0.027623026926648097, 0.12387805632930982, 0.12070566388115135, 0.0, 0.4504023522129372]),
    'MEADDataset': torch.tensor([0.13529189257344598, 0.13573319883999496, 0.267904425671416, 0.1339049300214349, 0.13459841129744043, 0.13217122683142102, 0.0, 0.060395914764846806]),
    'CaFEDataset': torch.tensor([0.15384615384615385, 0.15384615384615385, 0.15384615384615385, 0.15384615384615385, 0.15384615384615385, 0.15384615384615385, 0.0, 0.07692307692307693]),
    'ExpressoDataset': torch.tensor([0.2576543416429647, 0.12974736489877864, 0.002007696168646478, 0.0006692320562154927, 0.0006692320562154927, 0.0010875020913501756, 0.3962690312865986, 0.2118955997992304]),
    'ShEMODataset': torch.tensor([0.067, 0.14966666666666667, 0.0, 0.0, 0.075, 0.353, 0.012666666666666666, 0.3426666666666667]),
    'SUBESCODataset': torch.tensor([0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.0, 0.14285714285714285])
}
dataset_names_to_folders = {
    "URDUDataset": "URDU-Dataset-master",
    "EmoDBDataset": "emodb",
    "EMOVODataset": "EMOVO",
    "eNTERFACEDataset": "enterface database",
    "MESDDataset": "MESD",
    "MASCDataset": "MASC/data",
    "DEMoSDataset": "DEMoS/wav_DEMoS/",
    "CASIADataset": "CASIA/CASIA/6/",
    "AESDDDataset": "AESDD",
    "BAUMDataset": "BAUM1",
    "EEKKDataset": "EEKK",
    "ThorstenDataset": "thorsten-emotional_v02",
    "RESDDataset": "RESD",
    "MELDDataset": "MELD/MELD.Raw",
    "MEADDataset": "MEAD/audiofolder/audio",
    "CaFEDataset": "CaFE_48k",
    "ExpressoDataset": "expresso/audio_48khz",
    "ShEMODataset": "ShEMO",
    "SUBESCODataset": "SUBESCO"
}
BASE_MODEL_CONFIG = {
    "facebook/w2v-bert-2.0": {
        "model_class": Wav2Vec2BertModel,
        "processor_class": SeamlessM4TFeatureExtractor,
        "downsampling_factor": 320,
    },
    "facebook/wav2vec2-large-robust": {
        "model_class": Wav2Vec2Model,
        "processor_class": Wav2Vec2FeatureExtractor,
        "downsampling_factor": 320,
    },
    "facebook/hubert-large-ll60k": {
        "model_class": HubertModel,
        "processor_class": Wav2Vec2FeatureExtractor,
        "downsampling_factor": 320,
    },
    "microsoft/wavlm-large": {
        "model_class": WavLMModel,
        "processor_class": Wav2Vec2FeatureExtractor,
        "downsampling_factor": 320,
    },
    "laion/larger_clap_music_and_speech": {
        "model_class": ClapModel,
        "processor_class": ClapFeatureExtractor,
        "downsampling_factor": 320,
    },
    "m-a-p/MERT-v1-330M": {
        "model_class": AutoModel,
        "processor_class": Wav2Vec2FeatureExtractor,
        "downsampling_factor": 320,
    },
    "openai/whisper-medium": {
        "model_class": WhisperFlashAttentionForConditionalGeneration,
        "processor_class": WhisperProcessor,
        "downsampling_factor": 320,
    },
    "openai/whisper-large-v2": {
        "model_class": WhisperFlashAttentionForConditionalGeneration,
        "processor_class": WhisperProcessor,
        "downsampling_factor": 320,
    },
    "openai/whisper-large-v3": {
        "model_class": WhisperFlashAttentionForConditionalGeneration,
        "processor_class": WhisperProcessor,
        "downsampling_factor": 320,
    },
    "openai/whisper-large": {
        "model_class": WhisperFlashAttentionForConditionalGeneration,
        "processor_class": WhisperProcessor,
        "downsampling_factor": 320,
    },
}


DATASET_CLUSTERS = {
    "cluster1": {
        "datasets": ["EmoDBDataset", "RESDDataset"],
        "num_classes": 6,
        "labels": ["neutral", "anger", "fear", "happiness", "sadness", "disgust"]
    },
    "cluster2": {
        "datasets": ["EMOVODataset", "MESDDataset", "DEMoSDataset", "BAUMDataset", "MELDDataset", "MEADDataset", "CaFEDataset", "ExpressoDataset", "SUBESCODataset"],
        "num_classes": 7,
        "labels": ["neutral", "anger", "surprise", "fear", "happiness", "sadness", "disgust"]
    },
    "cluster3": {
        "datasets": ["URDUDataset", "EEKKDataset"],
        "num_classes": 4,
        "labels": ["happiness", "neutral", "sadness", "anger"]
    }
}


def clean_string(string):
    return ''.join(e for e in string if e.isalnum()).lower()


GLOBAL_LABEL_MAP = {
    'happiness': 0, 'sadness': 1, 'disgust': 2, 'fear': 3,
    'surprise': 4, 'anger': 5, 'other': 6, 'neutral': 7
}