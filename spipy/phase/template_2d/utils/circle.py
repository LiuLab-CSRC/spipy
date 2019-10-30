import numpy as np

def make_beamstop(shape, rad, is_fft_shifted = True):
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    rs      = np.sqrt((i**2 + j**2).astype(np.float))
        
    if is_fft_shifted is False :
        rs = np.fft.fftshift(rs)

    return rs >= rad
