\# ADR-0002: Feature extraction contract between training and inference



\## Status

Draft



\## Context

The log-mel spectrogram parameters used during training must be exactly

reproduced in the browser at inference time. Any mismatch breaks the model.



\## Decision

TBD — to be defined in Phase 1 with exact parameter values.



\## Consequences

\- These values will be frozen after Phase 1 and must not change.

