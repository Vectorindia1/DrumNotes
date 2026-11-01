# LICENSE_AUDIT (preliminary, not legal advice)

- **ADTOF code + weights: CC BY-NC-SA 4.0.** Non-commercial and share-alike. Cannot ship in a commercial release without permission from the authors. Any ADTOF-derived port (incl. ADTOF-pytorch) inherits this until proven otherwise.
- ADTOF-pytorch: no license file detected via the GitHub API; unlicensed code defaults to all rights reserved. Must be clarified with the author before any distribution.
- Demucs: repo MIT. Pretrained weights and training-data terms still to be checked individually.
- MuseScore: GPL. Calling the CLI as a separate process is the intended integration; revisit if bundled.
- Datasets (ENST Drums, IDMT-SMT-Drums, Slakh2100, MDBDrums): not yet investigated.

Implication: the prototype can use ADTOF for development, but a commercial release needs either permission or a replacement model (e.g. a self-trained model on licensed data, or a non-ML onset/classifier baseline).
