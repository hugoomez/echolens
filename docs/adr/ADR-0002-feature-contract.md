\# ADR-0002: Feature extraction contract between training and inference



\## Status

Accepted



\## Context

The log-mel spectrogram parameters used during training must be exactly

reproduced in the browser at inference time. Any mismatch between the

Python and JavaScript implementations breaks the model, since it would

see input statistically different from what it learned on.



\## Decision

Freeze the following parameters, used identically on both sides:



| Parameter | Value |

|---|---|

| Sample rate (SR) | 22050 Hz |

| FFT size (N\_FFT) | 1024 |

| Hop length (HOP) | 512 |

| Mel bands (N\_MELS) | 64 |

| Frequency range | 0 – 11025 Hz (SR/2) |

| Clip length | 4 s (88200 samples), pad/truncate |

| STFT window | Hann, `center=True` |

| Mel filterbank | Slaney normalization, `htk=False` |

| Log compression | `log(mel + 1e-6)` |



The mel filterbank matrix itself is precomputed once in Python and exported

verbatim to `web/public/models/mel\_filterbank.json`, so the browser never

recomputes it — it only performs the matrix multiplication against this

fixed matrix. This removes the single biggest source of cross-language

numerical drift (see risk R2 in the project's risk register).



Normalization statistics (global mean and standard deviation of the

training log-mel values) are computed at training time, not hardcoded,

and exported to `web/public/models/norm\_stats.json`.



\## Alternatives considered

\- Recomputing the mel filterbank independently in JavaScript: rejected,

&#x20; since even small floating-point or rounding differences between

&#x20; implementations would shift the model's input distribution.



\## Consequences

\- Any future change to these parameters requires: re-exporting the

&#x20; filterbank, re-training the model, and updating the JS feature

&#x20; extractor in Phase 3 to match.

\- `center=True` must be replicated exactly in the JS STFT implementation;

&#x20; inconsistency here causes frame misalignment (see "Errores comunes" in

&#x20; the Phase 1 plan).

