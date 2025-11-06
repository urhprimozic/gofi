# AdamEVE
## Adaptive Moment Estimation with Enhanced Velocity for Exploration

**AdamEVE** (Adam + Escape Velocity Enhancement) je razširitev standardnega Adam optimizatorja v PyTorchu, ki dodaja stohastični "escape momentum" za pobeg iz saddle točk in plitvih lokalnih minimumov.

## Glavne značilnosti


- Kombinira klasičen **Adam** update z dodatnim, stohastičnim momentom, ki pomaga pri premikanju iz stagnirajočih točk.
- Escape momentum je definiran kot:  
  $$
  b_n = a_n + \alpha_n \tilde{a}_{n-1}
  $$
  kjer:  
  - $a_n$ je standardni Adam update.  
  - $\tilde{a}_{n-1}$ je naključna perturbacija prejšnjega update-a $a_{n-1}$, vzorčena iz normalne porazdelitve.  
  - $\alpha_n = \max(1, \text{loss}) \cdot \max(1, f(\|\text{grad}\|))$, kjer $f(\|\text{grad}\|)$ zmanjšuje escape faktor, če je gradient velik.

- Enostavna integracija v obstoječe PyTorch kode, kompatibilna z `loss.backward()` in standardnim training loop-om.
- Escape momentum `b_n = a_n + α_n * ã_{n-1}`:
  - `a_n` je standardni Adam update.
  - `ã_{n-1}` je naključna perturbacija prejšnjega update-a (`a_{n-1}`), vzorčena iz normalne porazdelitve.
  - `α_n` je faktor, ki se povečuje, ko smo blizu lokalnega minimuma ali saddle točke:  
    `α_n = max(1, loss) * max(1, f(||grad||))`
- Funkcija `f(||grad||)` zmanjšuje escape faktor, če je gradient velik, kar preprečuje nepotrebno motnjo v močnih gradiencih.
- Enostavna integracija v obstoječe PyTorch kode, kompatibilna z `loss.backward()` in standardnim training loop-om.

## Parametri

| Parameter       | Opis                                                                 |
|-----------------|----------------------------------------------------------------------|
| `lr`            | Learning rate (default: 1e-3)                                        |
| `betas`         | Adam betas (default: (0.9, 0.999))                                   |
| `eps`           | Small epsilon to avoid division by zero (default: 1e-8)              |
| `gamma`         | Scaling factor for stochastic perturbation (default: 0.1)            |
| `f_max`         | Maximum scaling for gradient-based α (default: 10.0)                 |
| `p`             | Power of gradient norm in f(||grad||) (default: 1.0)                 |
| `weight_decay`  | Weight decay / L2 regularization (default: 0.0)                      |

## Primer uporabe

```python
import torch
from adameve import AdamEVE  # assuming this file is adameve.py

model = MyModel()
optimizer = AdamEVE(model.parameters(), lr=1e-3, gamma=0.05)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step(loss=loss)
