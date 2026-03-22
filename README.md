# Inversion of Induced Polarization Parameters from Time-Domain Electromagnetic Data Using the Debye Decomposition Model

**Masayuki Motoori¹², Lindsey J. Heagy¹, Gosuke Hoshino², Kunpei Nagase², Takumi Sato²**  
¹ University of British Columbia Geophysical Inversion Facility  
² Japan Organization for Metals and Energy Security (JOGMEC)

---

## Abstract

Time-domain electromagnetics (TEM) is a valuable tool for exploring seafloor hydrothermal deposits, which are characterized by variations in resistivity and chargeability. Several surveys using the Waseda Integrated Seafloor Time-domain EM (WISTEM) system have been conducted. Negative transients, attributed to induced polarization (IP) effects, were observed in data collected over a known deposit in the Okinawa Trough (2018).

The Cole–Cole model is commonly used to parameterize complex resistivity due to IP effects. However, it implicitly assumes sensitivity to a wide range of frequencies. In practice, TEM data have limited frequency content, constrained by the measurement time range and the diffusive nature of electromagnetic fields. Using a synthetic example, we demonstrate that inversion with the Cole–Cole model can become unstable when the data do not sufficiently constrain all four parameters.

The Debye Decomposition model addresses this limitation by enabling explicit selection of a relaxation time band. This allows separation of chargeability effects from resistivity influences in the data. In this work, we present a workflow for inverting TEM data using the Debye Decomposition model, including strategies for selecting an appropriate relaxation time band.

We compare inversion results from TEM data with those obtained from spectral induced polarization (SIP) data using the same Debye framework. By applying a consistent relaxation time band across both datasets, we ensure comparability of recovered IP parameters — a process we refer to as reconciliation. Synthetic examples demonstrate consistency between TEM inversion results and reconciled SIP-derived models.

Finally, we present a field application showing that the method successfully recovers IP parameters consistent with expected physical properties of seafloor hydrothermal deposits.

---

## Keywords

- Electromagnetics
- Inversion
- Data Processing
- Induced Polarization
- Marine Geophysics

---

## Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| EM | Electromagnetic methods |
| TEM | Time-domain Electromagnetics |
| IP | Induced Polarization |
| SIP | Spectral Induced Polarization |
| WISTEM | Waseda Integrated Seafloor Time-domain EM |

## References

- Bloomer, S., Kowalczyk, M., Kowalczyk, P., Constable, S., Haber, E., & Kasuga, T. (2018).  
  *AUV-CSEM: An improvement in the efficiency of multi-sensor mapping of seafloor massive sulfide (SMS) deposits with an AUV.*  
  In OCEANS 2018 MTS/IEEE Kobe Techno-Oceans (OTO), 1–7.  
  https://ieeexplore.ieee.org/document/8559049/

- Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., & Oldenburg, D. W. (2015).  
  *SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications.*  
  Computers & Geosciences, 85, 142–154.  
  https://linkinghub.elsevier.com/retrieve/pii/S009830041530056X

- Commer, M., Petrov, P. V., & Newman, G. A. (2017).  
  *FDTD modeling of induced polarization phenomena in transient electromagnetics.*  
  Geophysical Journal International.  
  https://doi.org/10.1093/gji/ggx023

- Couto Junior, M. A., Fiandaca, G., Maurya, P. K., Christiansen, A. V., Porsani, J. L., & Auken, E. (2020).  
  *AEMIP robust inversion using maximum phase angle Cole–Cole model re-parameterisation.*  
  Exploration Geophysics, 51, 170–183.  
  https://doi.org/10.1080/08123985.2019.1682458

- Endo, M., Nakayama, K., & Saito, A. (2024).  
  *Effective and practical interpretation of marine transient electromagnetic data (2): Quantitative interpretation.*  
  Proceedings of the SEGJ Conference.

- Fiandaca, G., Auken, E., Christiansen, A. V., & Gazoty, A. (2012).  
  *Time-domain induced polarization: Full-decay forward modeling and 1D inversion of Cole–Cole parameters.*  
  Geophysics, 77, E213–E225.  
  https://doi.org/10.1190/geo2011-0217.1

- Fiandaca, G., Madsen, L. M., & Maurya, P. K. (2018).  
  *Re-parameterisations of the Cole–Cole model for improved spectral inversion of IP data.*  
  Near Surface Geophysics, 16, 385–399.  
  https://doi.org/10.3997/1873-0604.2017065

- Fitterman, D. V., & Anderson, W. L. (1987).  
  *Effect of transmitter turn-off time on transient soundings.*  
  Geoexploration, 24, 131–146.

- Hase, J., Gurin, G., Titov, K., & Kemna, A. (2023).  
  *Conversion of induced polarization data from time to frequency domain using Debye decomposition.*  
  Minerals, 13, 955.  
  https://doi.org/10.3390/min13070955

- Heagy, L. J., Cockett, R., Kang, S., Rosenkjaer, G. K., & Oldenburg, D. W. (2017).  
  *A framework for simulation and inversion in electromagnetics.*  
  Computers & Geosciences, 107, 1–19.

- Kang, S. (2018).  
  *On recovering distributed induced polarization information from TEM data.*  
  PhD Thesis, University of British Columbia.

- Kang, S., & Oldenburg, D. W. (2016).  
  *Recovering distributed IP information from time-domain EM data.*  
  Geophysical Journal International, 207, 174–196.

- Kang, S., Oldenburg, D. W., & Heagy, L. J. (2020).  
  *Detecting induced polarization effects in time-domain data.*  
  Exploration Geophysics, 51, 122–133.

- Kemna, A., et al. (2012).  
  *Overview of spectral induced polarization method for near-surface applications.*  
  Near Surface Geophysics, 10, 453–468.

- Komori, S., et al. (2017).  
  *Depth profiles of resistivity and spectral IP for submarine hydrothermal deposits.*  
  Earth, Planets and Space, 69, 114.

- Marchant, D., Haber, E., & Oldenburg, D. W. (2014).  
  *3D modeling of IP effects in time-domain EM data.*  
  Geophysics, 79, E303–E314.

- Motoori, M., Heagy, L., et al. (2025).  
  *1D inversion of TEM data with IP effects for seafloor hydrothermal deposits.*  
  NSG 2025 Conference.

- Motoori, M., & Heagy, L. J. (2024).  
  *Synthetic study of IP effects on TEM data.*  
  AGU.

- Nakayama, K., & Saito, A. (2016).  
  *Practical marine TDEM systems using ROV.*  
  IEEE Techno-Ocean.

- Oldenburg, D. W., & Li, Y. (1994).  
  *Inversion of induced polarization data.*  
  Geophysics, 59, 1327–1341.

- Pelton, W. H., et al. (1978).  
  *Mineral discrimination with multifrequency IP.*  
  Geophysics, 43, 588–609.

- Tarasov, A., & Titov, K. (2007).  
  *Relaxation time distribution from IP measurements.*  
  Geophysical Journal International, 170, 31–43.

- Weigand, M., & Kemna, A. (2016).  
  *Debye decomposition of spectral IP data.*  
  Computers & Geosciences, 86, 34–45.

- Werthmüller, D. (2017).  
  *Empymod: Open-source 3D electromagnetic modeler.*  
  Geophysics, 82, WB9–WB19.  
  https://doi.org/10.1190/geo2016-0626.1

- Zhdanov, M. S., & Xu, Z. (2015).  
  *3D Cole–Cole inversion of IP data.*  
  IEEE GRSL, 12, 1180–1184.
| ROV | Remotely Operated Vehicle |
| CTD | Conductivity, Temperature, and Depth |
