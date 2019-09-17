from data.audio.process_audio import process_audio
from torchaudio.transforms import MuLawEncoding

process_audio('/media/jan//Data/datasets/PianoDataset',
              transforms=[MuLawEncoding(quantization_channels=256)], downsample=8000)