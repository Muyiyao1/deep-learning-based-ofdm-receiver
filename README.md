# Deep Learning Based OFDM Receiver

[Chinese README](README.zh-CN.md)

This repository implements a reproducible OFDM receiver simulation platform for comparing traditional pilot-based receivers with a CNN-based channel-estimation receiver. The project is organized as a Python codebase rather than a single notebook: it includes OFDM system modeling, modulation and channel utilities, baseline receivers, PyTorch training code, evaluation scripts, tests, and result visualizations.

## Project Background

Orthogonal Frequency Division Multiplexing (OFDM) is widely used in wireless systems such as Wi-Fi, LTE, and 5G NR. A conventional receiver estimates the channel from known pilot subcarriers, interpolates the channel response over data subcarriers, and then applies equalization before hard-decision demodulation.

This project studies a hybrid receiver design. The CNN does not replace the whole receiver. Instead, it learns to reconstruct the full frequency-domain channel response from sparse LS pilot estimates and a pilot mask. The estimated channel is then used by standard ZF or MMSE equalization.

## System Model

Default OFDM settings:

| Parameter | Value |
| --- | --- |
| Subcarriers | `64` |
| Cyclic prefix length | `16` |
| OFDM symbols per frame | `14` |
| Default modulation | QPSK |
| Pilot type | comb-type pilots |
| Default pilot spacing | `4` |
| Channel model | frequency-selective Rayleigh multipath |
| Default channel taps | `8` |
| Evaluation SNRs | `0, 5, 10, 15, 20, 25, 30 dB` |

The harder included experiment uses 16QAM, pilot spacing `8`, and a 12-tap Rayleigh channel to make the channel-estimation gap easier to observe.

## Traditional Baselines

The receiver is compared against pilot-based baselines:

- LS channel estimation on pilot subcarriers;
- linear interpolation for non-pilot subcarriers;
- ZF equalization;
- MMSE equalization.

These baselines are implemented as transparent communication-system references rather than black-box learning models.

## Deep Learning Method

The learning component is a CNN channel estimator implemented in PyTorch.

Input tensor:

```text
[B, 3, T, N]
```

The three input channels are sparse LS real part, sparse LS imaginary part, and pilot mask.

Output tensor:

```text
[B, 2, T, N]
```

The two output channels are the real and imaginary parts of the full channel response. The training objective is MSE over the complex channel response, represented as separate real and imaginary tensors.

Core files:

| Path | Purpose |
| --- | --- |
| `src/ofdm.py` | OFDM resource grid, modulation, cyclic prefix, and FFT/IFFT helpers. |
| `src/channel.py` | Rayleigh multipath channel and AWGN utilities. |
| `src/estimators.py` | LS interpolation and equalization helpers. |
| `src/models.py` | CNN channel-estimator model. |
| `src/train.py` | Training loop and checkpoint export. |
| `src/evaluate.py` | BER, SER, EVM, and channel-MSE evaluation. |

## Experimental Setup

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run traditional baselines:

```powershell
python scripts/run_baseline.py --num-test-frames 200
```

Train the CNN receiver component:

```powershell
python scripts/train_cnn_receiver.py --epochs 2 --num-train-samples 1000 --num-val-samples 200
```

Evaluate all methods after a checkpoint exists:

```powershell
python scripts/evaluate_all.py --num-test-frames 200
```

Run the included harder 16QAM sparse-pilot experiment:

```powershell
python scripts/run_challenging_experiment.py
```

Run tests:

```powershell
python -m pytest tests/
```

## Result Figures

Included result files are stored under `results/challenging/`.

![Challenging BER comparison](results/challenging/ber_comparison.png)

![Challenging SER comparison](results/challenging/ser_comparison.png)

![Channel MSE comparison](results/challenging/channel_mse_vs_snr.png)

![CNN gain vs SNR](results/challenging/cnn_gain_vs_snr.png)

In the included 16QAM sparse-pilot setting, CNN+MMSE reduces BER by about 24.8% at 20 dB compared with LS+MMSE, and the channel-MSE reduction is about 43.5%.

## Conclusion

The results show that the CNN channel estimator is most useful when pilots are sparse, modulation is more sensitive to channel-estimation error, and the channel is strongly frequency-selective. The project should be read as a modular learning-assisted OFDM receiver study: the CNN improves the channel-estimation component while the receiver still uses interpretable conventional equalization.

## Improvements

Possible next steps:

- compare against stronger traditional estimators such as LMMSE, DFT denoising, and spline interpolation;
- add time-varying channels, Doppler, CFO, phase noise, and hardware impairments;
- extend evaluation to 64QAM and coded BER or BLER;
- add confidence intervals for Monte Carlo results;
- test larger CNN, U-Net, or transformer-style channel estimators.

## Project Structure

```text
.
|-- README.md
|-- README.zh-CN.md
|-- requirements.txt
|-- report.pdf
|-- configs/
|   |-- default_config.json
|   `-- challenging_config.json
|-- data/
|-- src/
|   |-- channel.py
|   |-- config.py
|   |-- dataset.py
|   |-- estimators.py
|   |-- evaluate.py
|   |-- models.py
|   |-- modulation.py
|   |-- ofdm.py
|   |-- plots.py
|   |-- train.py
|   `-- utils.py
|-- scripts/
|   |-- run_baseline.py
|   |-- train_cnn_receiver.py
|   |-- evaluate_all.py
|   |-- run_challenging_experiment.py
|   `-- build_readme_pdf.py
|-- tests/
|-- notebooks/
`-- results/
    `-- challenging/
```
