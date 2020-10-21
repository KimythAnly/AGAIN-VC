import torch
import soundfile as sf

class VocoderWrapper():
    def __init__(self, device):
        self.device = device
        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        self.n_mels = 80
        self.sr = 22050

    def mel2wav(self, mel, save=''):
        device = self.device
        with torch.no_grad():
            if type(mel) is torch.Tensor:
                mel = mel.squeeze()
                mel = mel[None].to(device).float()
            else:
                mel = torch.from_numpy(mel[None]).to(device).float()
            y = self.vocoder.inverse(mel).cpu().numpy().flatten()
        if save != '':
            # librosa.output.write_wav(path=save, y=y, sr=sr)
            sf.write(file=save, data=y, samplerate=self.sr)
        return y


def get_vocoder(device):
    return VocoderWrapper(device=device)

