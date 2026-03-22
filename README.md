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
| ROV | Remotely Operated Vehicle |
| CTD | Conductivity, Temperature, and Depth |
