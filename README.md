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
## References

Binley, A. and Slater, L. (2020) Resistivity and Induced Polarization: Theory and Applications to the Near-Surface Earth. Cambridge
University Press, 1 edn. URL: https://www.cambridge.org/core/product/identifier/9781108685955/type/book.

Bloomer, S., Kowalczyk, M., Kowalczyk, P., Constable, S., Haber, E. and Kasuga, T. (2018) AUV-CSEM: An Improvement in the
Efficiency ofMulti-SensorMapping of SeafloorMassive Sulfide (SMS) Deposits with an AUV. In 2018 OCEANS - MTS/IEEE
Kobe Techno-Oceans (OTO), 1–7. Kobe: IEEE. URL: https://ieeexplore.ieee.org/document/8559049/.

Cockett, R., Kang, S.,Heagy, L. J., Pidlisecky, A. andOldenburg,D.W. (2015) SimPEG: An open source framework for simulation
and gradient based parameter estimation in geophysical applications. Computers & Geosciences, 85, 142–154. URL: https:
//linkinghub.elsevier.com/retrieve/pii/S009830041530056X.

Commer, M., Petrov, P. V. and Newman, G. A. (2017) FDTD modeling of induced polarization phenomena in transient electromagnetics.
Geophysical Journal International, ggx023. URL: https://academic.oup.com/gji/article-lookup/doi/10.
1093/gji/ggx023.

Couto Junior, M. A., Fiandaca, G.,Maurya, P. K., Christiansen, A. V., Porsani, J. L. and Auken, E. (2020) AEMIP robust inversion
using maximum phase angle Cole–Cole model re-parameterisation applied for HTEM survey over Lamego gold mine,
Quadrilátero Ferrífero, MG, Brazil. Exploration Geophysics, 51, 170–183. URL: https://www.tandfonline.com/doi/full/
10.1080/08123985.2019.1682458.

Endo, M., NAKAYAMA, K. and Saito, A. (2024) Effective and Practical Interpretation ofMarine Transient Electromagnetic Data
(2): Quantitative interpretation. In Proceedings of the SEGJ Conference.

Fiandaca, G., Auken, E., Christiansen, A. V. and Gazoty, A. (2012) Time-domain-induced polarization: Full-decay forward
modeling and 1D laterally constrained inversion of Cole-Cole parameters. GEOPHYSICS, 77, E213–E225. URL: https:
//library.seg.org/doi/10.1190/geo2011-0217.1.

Fiandaca, G., Madsen, L. M. and Maurya, P. K. (2018) Re-parameterisations of the Cole–Cole model for improved spectral
inversion of induced polarization data. Near Surface Geophysics, 16, 385–399. URL: https://onlinelibrary.wiley.com/
doi/10.3997/1873-0604.2017065.

Fitterman, D. V. and Anderson, W. L. (1987) Effect of transmitter turn-off time on transient soundings. Geoexploration, 24,
131–146. URL: https://linkinghub.elsevier.com/retrieve/pii/0016714287900871.

Hase, J., Gurin, G., Titov, K. and Kemna, A. (2023) Conversion of Induced Polarization Data and Their Uncertainty from Time
Domain to Frequency Domain Using Debye Decomposition. Minerals, 13, 955. URL: https://www.mdpi.com/2075-163X/
13/7/955.

Heagy, L. J., Cockett, R., Kang, S., Rosenkjaer, G. K. and Oldenburg, D. W. (2017) A framework for simulation and inversion
in electromagnetics. Computers & Geosciences, 107, 1–19. URL: https://linkinghub.elsevier.com/retrieve/pii/
S0098300416303946.

Japan Organization for Metals and Energy Security (2023) JOGMEC successfully identifies Mineral Resource Potential at 50-
million-ton level through resource assessment of seafloor hydrothermal deposits - Steady progress towards development
of seafloor hydrothermal deposits in EEZ zone -. URL: https://www.jogmec.go.jp/english/news/release/news_10_00051.
html.

Kang, S. (2018) On recovering distributed induced polarization information from time-domain electromagnetic data. Ph.D. thesis,
University of British Columbia. URL: https://doi.library.ubc.ca/10.14288/1.0364164. Version Number: 1.

Kang, S., Fournier, D. andOldenburg, D.W. (2017) Inversion of airborne geophysics over the DO-27/DO-18 kimberlites—Part
3: Induced polarization. Interpretation, 5, T327–T340. URL: https://library.seg.org/doi/10.1190/INT-2016-0141.1.

Kang, S. and Oldenburg, D. W. (2016) On recovering distributed IP information from inductive source time domain electromagnetic
data. Geophysical Journal International, 207, 174–196. URL: https://academic.oup.com/gji/articlelookup/
doi/10.1093/gji/ggw256.

— (2019) Inversions of time-domain spectral induced polarization data using stretched exponential. Geophysical Journal International,
219, 1851–1865. URL: https://academic.oup.com/gji/article/219/3/1851/5567618.

Kang, S., Oldenburg, D. W. and Heagy, L. J. (2020) Detecting induced polarisation effects in time-domain data: a modelling
study using stretched exponentials. Exploration Geophysics, 51, 122–133. URL: https://www.earthdoc.org/content/
journals/10.1080/08123985.2019.1690393.

Kemna, A., Binley, A., Cassiani, G., Niederleithinger, E., Revil, A., Slater, L., Williams, K. H., Orozco, A. F., Haegel, F., Hördt, A.,
Kruschwitz, S., Leroux, V., Titov, K. and Zimmermann, E. (2012) An overview of the spectral induced polarization method
for near-surface applications. Near Surface Geophysics, 10, 453–468. URL: https://onlinelibrary.wiley.com/doi/10.
3997/1873-0604.2012027.

Komori, S., Masaki, Y., Tanikawa, W., Torimoto, J., Ohta, Y., Makio, M., Maeda, L., Ishibashi, J.-i., Nozaki, T., Tadai, O. and
Kumagai, H. (2017) Depth profiles of resistivity and spectral IP for active modern submarine hydrothermal deposits: a
case study from the Iheya North Knoll and the Iheya Minor Ridge in Okinawa Trough, Japan. Earth, Planets and Space, 69,
114. URL: http://earth-planets-space.springeropen.com/articles/10.1186/s40623-017-0691-6.

Kowalczyk, P., Bloomer, S. and Kowalczyk, M. (2015) Geophysical Methods for the Mapping of Submarine Massive Sulphide
Deposits. In Offshore Technology Conference, OTC–26049–MS. Houston, Texas, USA: OTC. URL: https://onepetro.org/
OTCONF/proceedings/15OTC/15OTC/OTC-26049-MS/77879.

Kratzer, T. and Macnae, J. C. (2012) Induced polarization in airborne EM. GEOPHYSICS, 77, E317–E327. URL: https://
library.seg.org/doi/10.1190/geo2011-0492.1.

MAKUUCHI, A., MATSUKUMA, Y., AMEMIYA, Y., IYATOMI, N., SASAKI, Y., ICHII, Y., TAKAHASHI, T., TAKAKURA, S., SASAKI,
Y., UEDA, S., MOTOORI, M. and MASUDA, K. (2017) Application of the SIP method to metal resource exploration. Shigen-
Chishitsu, 67, 5–18. URL: https://www.jstage.jst.go.jp/article/shigenchishitsu/67/1/67_5/_article/-char/en.

Marchant,D.,Haber, E. andOldenburg,D.W. (2014) Three-dimensional modeling of IP effects in time-domain electromagnetic
data. GEOPHYSICS, 79, E303–E314. URL: https://library.seg.org/doi/10.1190/geo2014-0060.1.

MOROZUMI, H.,WATANABE, K., SAKURAI, H., HINO, H., KADO, Y., MOTOORI, M. and TENDO, H. (2020) Additional information
for characteristics of seafloor hydrothermal deposits investigated by JOGMEC. Shigen-Chishitsu, 70, 113–119.

Motoori, M., Heagy, L., Hoshino, G., Yamamoto, K., Morozumi, H., Nagase, K. and Sugimoto, S. (2025) 1D Inversion of Time-
Domain Electromagnetic Data with Induced Polarization Effects for a Sea-Floor Hydrothermal Deposit. In NSG 2025: 6th
Conference on Geophysics for Mineral Exploration and Mining, 1–5. Naples, Italy,: European Association of Geoscientists &
Engineers. URL: https://www.earthdoc.org/content/papers/10.3997/2214-4609.202520037.

Motoori, M. and Heagy, L. J. (2024) A synthetic study investigating induced polarization effects on time-domain electromagnetic
data for Sea-floor hydrothermal deposit. vol. 2024, NS43C–03. URL: https://ui.adsabs.harvard.edu/abs/
2024AGUFMNS43C..03M. ADS Bibcode: 2024AGUFMNS43C..03M.

Müller, H., Schwalenberg, K., Reeck, K., Barckhausen, U., Schwarz-Schampera, U., Hilgenfeldt, C. and Von Dobeneck, T. (2018)
Mapping seafloor massive sulfides with the Golden Eye frequency-domain EM profiler. First Break, 36, 61–67. URL:
https://www.earthdoc.org/content/journals/10.3997/1365-2397.n0127.

Nakayama, K. (2016) Study on theMarine TDEM for the Ocean Bottom Hydrothermal Deposits. Doctoral Thesis,Waseda University,
Tokyo, Japan. URL: http://hdl.handle.net/2065/00054603.

Nakayama, K.,Moroori, M. and Saito, A. (2019) Application of Time-Domain Electromagnetic Survey for Seafloor Polymetallic
Sulphides in the Okinawa Trough. In 25th European Meeting of Environmental and Engineering Geophysics, 1–5. The Hague,
Netherlands,: European Association of Geoscientists & Engineers. URL: https://www.earthdoc.org/content/papers/10.
3997/2214-4609.201902383.

Nakayama, K. and Saito, A. (2016) Practical marine TDEM systems using ROV for the ocean bottom hydrothermal deposits.
In 2016 Techno-Ocean (Techno-Ocean), 643–647. Kobe, Japan: IEEE. URL: http://ieeexplore.ieee.org/document/
7890734/.

Nordsiek, S. andWeller, A. (2008) A new approach to fitting induced-polarization spectra. GEOPHYSICS, 73, F235–F245. URL:
https://library.seg.org/doi/10.1190/1.2987412.

Oldenburg, D. W. and Li, Y. (1994) Inversion of induced polarization data. GEOPHYSICS, 59, 1327–1341. URL: https://
library.seg.org/doi/10.1190/1.1443692.
— (2005) Inversion for Applied Geophysics: A Tutorial. In Near-Surface Geophysics. Society of Exploration Geophysicists. URL:
https://doi.org/10.1190/1.9781560801719.ch5.

Pelton, W. H., Ward, S. H., Hallof, P. G., Sill, W. R. and Nelson, P. H. (1978) MINERAL DISCRIMINATION AND REMOVAL
OF INDUCTIVE COUPLING WITH MULTIFREQUENCY IP. GEOPHYSICS, 43, 588–609. URL: https://library.seg.org/
doi/10.1190/1.1440839.

Reeck, K., Müller, H., Hölz, S., Haroon, A., Schwalenberg, K. and Jegen, M. (2020) Effects of metallic system components on
marine electromagnetic loop data. Geophysical Prospecting, 68, 2254–2270. URL: https://onlinelibrary.wiley.com/
doi/10.1111/1365-2478.12984.

Safipour, R., Hölz, S., Jegen, M. and Swidinsky, A. (2018) A first application of a marine inductive source electromagnetic
configuration with remote electric dipole receivers: Palinuro Seamount, Tyrrhenian Sea. Geophysical Prospecting, 66, 1415–
1432. URL: https://onlinelibrary.wiley.com/doi/10.1111/1365-2478.12646.

Spagnoli, G., Hannington, M., Bairlein, K., Hördt, A., Jegen, M., Petersen, S. and Laurila, T. (2016) Electrical properties of
seafloor massive sulfides. Geo-Marine Letters, 36, 235–245. URL: http://link.springer.com/10.1007/s00367-016-0439-
5.

Takakura, S., Sasaki, Y., Takahashi, T. and Matsukuma, Y. (2014) Complex resistivity measurements of artificial samples containing
pyrite and magnetite particles. BUTSURI-TANSA(Geophysical Exploration), 67, 267–275. URL: https://www.jstage.
jst.go.jp/article/segj/67/4/67_267/_article/-char/ja/.

Tarasov, A. and Titov, K. (2007) Relaxation time distribution from time domain induced polarization measurements. Geophysical
Journal International, 170, 31–43. URL: https://academic.oup.com/gji/article-lookup/doi/10.1111/j.1365-
246X.2007.03376.x.
— (2013) On the use of the Cole–Cole equations in spectral induced polarization. Geophysical Journal International, 195,
352–356. URL: https://academic.oup.com/gji/article/195/1/352/608470.

Telford,W., Geldart, L. and Sheriff, R. (1990) 9. Induced Polarization. In Applied Geophysics (2nd Edition). Cambridge University
Press. URL: https://app.knovel.com/hotlink/pdf/id:kt007U1EZ1/applied-geophysics-2nd/induced-polarization.

Uhlmann, D. and Hakim, R. (1971) Derivation of distribution functions from relaxation data. Journal of Physics and Chemistry
of Solids, 32, 2652–2655. URL: https://linkinghub.elsevier.com/retrieve/pii/S0022369771801142.

Viezzoli, A., Kaminski, V. and Fiandaca, G. (2017)Modeling induced polarization effects in helicopter time domain electromagnetic
data: Synthetic case studies. GEOPHYSICS, 82, E31–E50. URL: https://library.seg.org/doi/10.1190/geo2016-
0096.1.

Weigand, M. and Kemna, A. (2016) Debye decomposition of time-lapse spectral induced polarisation data. Computers &
Geosciences, 86, 34–45. URL: https://linkinghub.elsevier.com/retrieve/pii/S0098300415300625.

Werthmüller, D. (2017) An open-source full 3D electromagnetic modeler for 1D VTI media in Python: empymod. GEOPHYSICS,
82, WB9–WB19. URL: https://library.seg.org/doi/10.1190/geo2016-0626.1.

Yuval and Oldenburg, D. W. (1997) Computation of Cole-Cole parameters from IP data. Geophysics, 62, 436–
448. URL: https://pubs.geoscienceworld.org/geophysics/article/62/2/436/73052/Computation-of-Cole-Coleparameters-
from-IP-data.

Zhengwei Xu and Zhdanov, M. S. (2015) Three-Dimensional Cole-Cole Model Inversion of Induced Polarization Data Based
on Regularized Conjugate Gradient Method. IEEE Geoscience and Remote Sensing Letters, 12, 1180–1184. URL: http:
//ieeexplore.ieee.org/document/7031900/.

Zonge, K., Wynn, J. and Urquhart, S. (2005) Resistivity, Induced Polarization, and Complex Resistivity. In Near-Surface
Geophysics (ed. D. K. Butler), vol. 13, 0. Society of Exploration Geophysicists. URL: https://doi.org/10.1190/1.


