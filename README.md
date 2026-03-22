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

[1] Werthmüller, Dieter (2017). An open-source full {3D} electromagnetic modeler for {1D} {VTI} media in {Python}: empymod. https://doi.org/10.1190/geo2016-0626.1
[2] Nakayama, Keiko and Saito, Akira (2016). Practical marine {TDEM} systems using {ROV} for the ocean bottom hydrothermal deposits. https://doi.org/10.1109/Techno-Ocean.2016.7890734
[3] Kang, Seogi and Oldenburg, Douglas W. (2016). On recovering distributed {IP} information from inductive source time domain electromagnetic data. https://doi.org/10.1093/gji/ggw256
[4] Heagy, Lindsey J. and Cockett, Rowan and Kang, Seogi and Rosenkjaer, Gudni K. and Oldenburg, Douglas W. (2017). A framework for simulation and inversion in electromagnetics. https://doi.org/10.1016/j.cageo.2017.06.018
[5] Cockett, Rowan and Kang, Seogi and Heagy, Lindsey J. and Pidlisecky, Adam and Oldenburg, Douglas W. (2015). {SimPEG}: {An} open source framework for simulation and gradient based parameter estimation in geophysical applications. https://doi.org/10.1016/j.cageo.2015.09.015
[6] Pelton, W. H. and Ward, S. H. and Hallof, P. G. and Sill, W. R. and Nelson, P. H. (1978). {MINERAL} {DISCRIMINATION} {AND} {REMOVAL} {OF} {INDUCTIVE} {COUPLING} {WITH} {MULTIFREQUENCY} {IP}. https://doi.org/10.1190/1.1440839
[7] Oldenburg, Douglas W. and Li, Yaoguo (2005). Inversion for {Applied} {Geophysics}: {A} {Tutorial}. https://doi.org/10.1190/1.9781560801719.ch5
[8] MOROZUMI, Haruhisa and WATANABE, Kazuo and SAKURAI, Hironobu and HINO, Hikari and KADO, Yasuyuki and MOTOORI, Masayuki and TENDO, Hiroko (2020). Additional information for characteristics of seafloor hydrothermal deposits investigated by {JOGMEC}. https://doi.org/10.11456/shigenchishitsu.70.113
[9] Tarasov, Andrey and Titov, Konstantin (2013). On the use of the {Cole}–{Cole} equations in spectral induced polarization. https://doi.org/10.1093/gji/ggt251
[10] Nordsiek, Sven and Weller, Andreas (2008). A new approach to fitting induced-polarization spectra. https://doi.org/10.1190/1.2987412
[11] Fitterman, David V and Anderson, Walter L (1987). Effect of transmitter turn-off time on transient soundings. https://doi.org/10.1016/0016-7142(87)90087-1
[12] Tarasov, Andrey and Titov, Konstantin (2007). Relaxation time distribution from time domain induced polarization measurements. https://doi.org/10.1111/j.1365-246X.2007.03376.x
[13] Endo, Masashi and NAKAYAMA, Keiko and Saito, Akira (2024). Effective and {Practical} {Interpretation} of {Marine} {Transient} {Electromagnetic} {Data} (2): {Quantitative} interpretation. https://doi.org/
[14] Fiandaca, Gianluca and Madsen, Line Meldgaard and Maurya, Pradip Kumar (2018). Re‐parameterisations of the {Cole}–{Cole} model for improved spectral inversion of induced polarization data. https://doi.org/10.3997/1873-0604.2017065
[15] Oldenburg, Douglas W. and Li, Yaoguo (1994). Inversion of induced polarization data. https://doi.org/10.1190/1.1443692
[16] Viezzoli, Andrea and Kaminski, Vladislav and Fiandaca, Gianluca (2017). Modeling induced polarization effects in helicopter time domain electromagnetic data: {Synthetic} case studies. https://doi.org/10.1190/geo2016-0096.1
[17] Marchant, David and Haber, Eldad and Oldenburg, Douglas W. (2014). Three-dimensional modeling of {IP} effects in time-domain electromagnetic data. https://doi.org/10.1190/geo2014-0060.1
[18] Spagnoli, Giovanni and Hannington, Mark and Bairlein, Katharina and Hördt, Andreas and Jegen, Marion and Petersen, Sven and Laurila, Tea (2016). Electrical properties of seafloor massive sulfides. https://doi.org/10.1007/s00367-016-0439-5
[19] Kowalczyk, Peter and Bloomer, Stephen and Kowalczyk, Matthew (2015). Geophysical {Methods} for the {Mapping} of {Submarine} {Massive} {Sulphide} {Deposits}. https://doi.org/10.4043/26049-MS
[20] Safipour, Roxana and Hölz, Sebastian and Jegen, Marion and Swidinsky, Andrei (2018). A first application of a marine inductive source electromagnetic configuration with remote electric dipole receivers: {Palinuro} {Seamount}, {Tyrrhenian} {Sea}. https://doi.org/10.1111/1365-2478.12646
[21] Reeck, Konstantin and Müller, Hendrik and Hölz, Sebastian and Haroon, Amir and Schwalenberg, Katrin and Jegen, Marion (2020). Effects of metallic system components on marine electromagnetic loop data. https://doi.org/10.1111/1365-2478.12984
[22] Motoori, M. and Heagy, L. and Hoshino, G. and Yamamoto, K. and Morozumi, H. and Nagase, K. and Sugimoto, S. (2025). {1D} {Inversion} of {Time}-{Domain} {Electromagnetic} {Data} with {Induced} {Polarization} {Effects} for a {Sea}-{Floor} {Hydrothermal} {Deposit}. https://doi.org/10.3997/2214-4609.202520037
[23] Motoori, Masayuki and Heagy, Lindsey Justine (2024). A synthetic study investigating induced polarization effects on time-domain electromagnetic data for {Sea}-floor hydrothermal deposit.. https://doi.org/
[24] Couto Junior, Marco Antonio and Fiandaca, Gianluca and Maurya, Pradip Kumar and Christiansen, Anders Vest and Porsani, Jorge Luís and Auken, Esben (2020). {AEMIP} robust inversion using maximum phase angle {Cole}–{Cole} model re-parameterisation applied for {HTEM} survey over {Lamego} gold mine, {Quadrilátero} {Ferrífero}, {MG}, {Brazil}. https://doi.org/10.1080/08123985.2019.1682458
[25] Fiandaca, Gianluca and Auken, Esben and Christiansen, Anders Vest and Gazoty, Aurélie (2012). Time-domain-induced polarization: {Full}-decay forward modeling and {1D} laterally constrained inversion of {Cole}-{Cole} parameters. https://doi.org/10.1190/geo2011-0217.1
[26] {Japan Organization for Metals and Energy Security} (2023). {JOGMEC} successfully identifies {Mineral} {Resource} {Potential} at 50-million-ton level through resource assessment of seafloor hydrothermal deposits - {Steady} progress towards development of seafloor hydrothermal deposits in {EEZ} zone -. https://doi.org/
[27] Kang, Seogi and Fournier, Dominique and Oldenburg, Douglas W. (2017). Inversion of airborne geophysics over the {DO}-27/{DO}-18 kimberlites — {Part} 3: {Induced} polarization. https://doi.org/10.1190/INT-2016-0141.1
[28] Nakayama, K. and Moroori, M. and Saito, A. (2019). Application of {Time}-{Domain} {Electromagnetic} {Survey} for {Seafloor} {Polymetallic} {Sulphides} in the {Okinawa} {Trough}. https://doi.org/10.3997/2214-4609.201902383
[29] Nakayama, Keiko (2016). Study on the {Marine} {TDEM} for the {Ocean} {Bottom} {Hydrothermal} {Deposits}. https://doi.org/
[30] Müller, Hendrik and Schwalenberg, Katrin and Reeck, Konstantin and Barckhausen, Udo and Schwarz-Schampera, Ulrich and Hilgenfeldt, Christian and Von Dobeneck, Tilo (2018). Mapping seafloor massive sulfides with the {Golden} {Eye} frequency-domain {EM} profiler. https://doi.org/10.3997/1365-2397.n0127
[31] MAKUUCHI, Ayumu and MATSUKUMA, Yuta and AMEMIYA, Yutaka and IYATOMI, Nobuyoshi and SASAKI, Yoji and ICHII, Yoshihiko and TAKAHASHI, Takeharu and TAKAKURA, Shinichi and SASAKI, Yutaka and UEDA, Satoshi and MOTOORI, Masayuki and MASUDA, Kazuo (2017). Application of the {SIP} method to metal resource exploration. https://doi.org/10.11456/shigenchishitsu.67.5
[32] Bloomer, Steve and Kowalczyk, Matthew and Kowalczyk, Peter and Constable, Steven and Haber, Eldad and Kasuga, Toru (2018). {AUV}-{CSEM}: {An} {Improvement} in the {Efficiency} of {Multi}-{Sensor} {Mapping} of {Seafloor} {Massive} {Sulfide} ({SMS}) {Deposits} with an {AUV}. https://doi.org/10.1109/OCEANSKOBE.2018.8559049
[33] Takakura, Shinichi and Sasaki, Yutaka and Takahashi, Takeharu and Matsukuma, Yuta (2014). Complex resistivity measurements of artificial samples containing pyrite and magnetite particles. https://doi.org/10.3124/segj.67.267
[34] Kratzer, Terence and Macnae, James C. (2012). Induced polarization in airborne {EM}. https://doi.org/10.1190/geo2011-0492.1
[35] Kang, Seogi (2018). On recovering distributed induced polarization information from time-domain electromagnetic data. https://doi.org/
[36] Binley, Andrew and Slater, Lee (2020). Resistivity and {Induced} {Polarization}: {Theory} and {Applications} to the {Near}-{Surface} {Earth}. https://doi.org/10.1017/9781108685955
[37] Weigand, M. and Kemna, A. (2016). Debye decomposition of time-lapse spectral induced polarisation data. https://doi.org/10.1016/j.cageo.2015.09.021
[38] Uhlmann, D.R. and Hakim, R.M. (1971). Derivation of distribution functions from relaxation data. https://doi.org/10.1016/S0022-3697(71)80114-2
[39] Yuval and Oldenburg, Douglas W. (1997). Computation of {Cole}-{Cole} parameters from {IP} data. https://doi.org/10.1190/1.1444154
[40] Zonge, Ken and Wynn, Jeff and Urquhart, Scott (2005). Resistivity, {Induced} {Polarization}, and {Complex} {Resistivity}. https://doi.org/10.1190/1.9781560801719.ch9
[41] Telford, W.M. and Geldart, L.P. and Sheriff, R.E. (1990). 9. {Induced} {Polarization}. https://doi.org/
[42] {Zhengwei Xu} and Zhdanov, Michael S. (2015). Three-{Dimensional} {Cole}-{Cole} {Model} {Inversion} of {Induced} {Polarization} {Data} {Based} on {Regularized} {Conjugate} {Gradient} {Method}. https://doi.org/10.1109/LGRS.2014.2387197
[43] Kemna, Andreas and Binley, Andrew and Cassiani, Giorgio and Niederleithinger, Ernst and Revil, André and Slater, Lee and Williams, Kenneth H. and Orozco, Adrián Flores and Haegel, Franz‐Hubert and Hördt, Andreas and Kruschwitz, Sabine and Leroux, Virginie and Titov, Konstantin and Zimmermann, Egon (2012). An overview of the spectral induced polarization method for near‐surface applications. https://doi.org/10.3997/1873-0604.2012027
[44] Commer, Michael and Petrov, Petr V. and Newman, Gregory A. (2017). {FDTD} modeling of induced polarization phenomena in transient electromagnetics. https://doi.org/10.1093/gji/ggx023
[45] Hase, Joost and Gurin, Grigory and Titov, Konstantin and Kemna, Andreas (2023). Conversion of {Induced} {Polarization} {Data} and {Their} {Uncertainty} from {Time} {Domain} to {Frequency} {Domain} {Using} {Debye} {Decomposition}. https://doi.org/10.3390/min13070955
[46] Komori, Shogo and Masaki, Yuka and Tanikawa, Wataru and Torimoto, Junji and Ohta, Yusuke and Makio, Masato and Maeda, Lena and Ishibashi, Jun-ichiro and Nozaki, Tatsuo and Tadai, Osamu and Kumagai, Hidenori (2017). Depth profiles of resistivity and spectral {IP} for active modern submarine hydrothermal deposits: a case study from the {Iheya} {North} {Knoll} and the {Iheya} {Minor} {Ridge} in {Okinawa} {Trough}, {Japan}. https://doi.org/10.1186/s40623-017-0691-6
[47] Kang, Seogi and Oldenburg, Douglas W (2019). Inversions of time-domain spectral induced polarization data using stretched exponential. https://doi.org/10.1093/gji/ggz401
[48] Kang, Seogi and Oldenburg, Douglas W. and Heagy, Lindsey J. (2020). Detecting induced polarisation effects in time-domain data: a modelling study using stretched exponentials. https://doi.org/10.1080/08123985.2019.1690393
[49] NAKAYAMA, Keiko and SAITO, Akira (2017). Processing of seabed {TDEM} data for the ocean bottom hydrothermal deposits. https://doi.org/

| ROV | Remotely Operated Vehicle |
| CTD | Conductivity, Temperature, and Depth |
