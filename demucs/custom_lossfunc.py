import torch
from torch import nn
import matplotlib.pyplot as plt
def torch_fft_hr(signal, N, sample_rate):
    y = signal
    T = 1.0 / sample_rate
    yf = torch.fft.fft(y, axis=-1)
    xf = torch.linspace(0.0, .5*sample_rate, N//2)
    spectrum = 2.0/N * torch.abs(yf[-1,:N//2])
    #only keep the heart rate frequency range
    # find the hr frequncy rang bins
    heart_band_index = (( (xf >=0.7)&(xf<=2.3) ).nonzero(as_tuple=True)[0])
    start = heart_band_index[0]
    end = heart_band_index[-1]
    #keep these
    spectrum = spectrum[..., start:end]
    xf = xf[..., start:end]
    #normalize
    spectrum_min,_ = torch.min(spectrum, dim=-1, keepdim=True)
    spectrum_max,_ = torch.max(spectrum, dim=-1, keepdim=True)
    spectrum_normalized = (spectrum - spectrum_min)/(spectrum_max - spectrum_min)
    return spectrum_normalized, xf
    #return spectrum, xf

def fft_l1loss(pred, target):
    pred_fft, ax = torch_fft_hr(pred, pred.shape[-1], 44100/70)
    print(pred_fft.shape)
    target_fft, ax = torch_fft_hr(target, target.shape[-1], 44100/70)
    print(target_fft.shape)

    plt.plot(ax.cpu().detach().numpy(), pred_fft.cpu().detach().numpy()[0,1,:], color="green")
    plt.plot(ax.cpu().detach().numpy(), target_fft.cpu().detach().numpy()[0,1,:], color="red")
    plt.show()

    L = nn.L1Loss()
    loss = L(pred_fft, target_fft)
    return loss
