# Sample datasets

The sample images used by the tests, examples, benchmarks and documentation figures are
not stored in this repository. They come from the `datasets/` folder of the original ONERA
GeFolki repository, [aplyer/gefolki](https://github.com/aplyer/gefolki/tree/3a77736109ef66470349fb82f2d04ae161a64850/datasets),
pinned to commit `3a77736` and checked by SHA-256.

```bash
python datasets/fetch.py                 # all files (~160 MB) into datasets/
python datasets/fetch.py WV.tif QB.tif   # some files
python datasets/fetch.py EvalGeFolki     # a folder (any path prefix)
python datasets/fetch.py --list          # names and sizes
```

Tests that need a missing file are skipped, with a message naming the file.

## Files

| file | content | size | used by |
|---|---|---|---|
| `WV.tif`, `QB.tif` | WorldView and QuickBird panchromatic, San Francisco, 2002 x 2802, uint16; QB pre-registered to WV with an affine fit on manual control points | 8 MB, 11 MB | quickstart, figures |
| `radar_bandep.png` | P-band airborne SAR (SETHI), Pauli RGB (R: HH-VV, G: HV+VH, B: HH+VV), NL-SAR filtered, Sweden, 2000 x 2000 | 7.7 MB | tests, benchmark, quickstart, figures |
| `optiquehr_georef.png` | RGB orthophoto (Lantmäteriet, 0.5 m) resampled onto the radar geometry | 6.8 MB | tests, quickstart, figures |
| `lidar_georef.png` | LIDAR DEM (SLU, 0.5 m) resampled onto the radar geometry | 2.3 MB | figures |
| `radar_bandel_hh1.mat`, `radar_bandel_hh2.mat` | L-band airborne SAR (SETHI) HH images, Matlab format, for SAR/SAR exercises | 37 MB, 46 MB | — |
| `S1_Jacksonville_GEE.tif` | Sentinel-1 GRD temporal mean (Google Earth Engine), Jacksonville, EPSG:4326 | 9.1 MB | `locate` tests, quickstart, figures |
| `JacksonvilleNavalAirStation_sandiaKu.png` | Ku-band airborne SAR chip, courtesy of Sandia National Laboratories, Radar ISR | 0.2 MB | `locate` tests, quickstart, figures |
| `S1_Washington_GEE.tif`, `WhashingtonDC_sandiaKu_project.png` | same pairing over Washington DC | 8.6 MB, 1.4 MB | — |
| `0414.png`, `4645.png` | small image chips from the original repository | < 20 kB | — |
| `EvalGeFolki/S1S2/` | Sentinel-1 (2 bands) and Sentinel-2 (3 bands) patches, 448 x 448, 10 m, EPSG:32631, and a ground-truth flow (`Flow_patch11.tif`, two pages: u, v) | 5.7 MB | accuracy tests, quickstart, figures |
| `EvalGeFolki/HR/` | high-resolution L-band Pauli radar (UAVSAR file naming, 1024 x 1024) and optical patches with a ground-truth flow | 15 MB | accuracy tests, quickstart |

The ground-truth flows hold two TIFF pages; GDAL sees one, so read them with `tifffile`.

## Sources and licences

All data are provided by the institutions below for testing GeFolki. Ask the providers
before using them for anything else.

- **LIDAR**: provided by Johan Fransson (SLU, Swedish University of Agricultural Sciences)
  and Lars Ulander (FOI, Swedish Defence Research Agency), processed by SLU from a
  0.5 m DEM. Acquisition funded by ESA through the
  [BioSAR 3 campaign](https://earth.esa.int/eogateway/campaigns/biosar-3).
- **SAR (Sweden)**: BioSAR 3 campaign, funded by ESA, acquired by the ONERA airborne SETHI
  system. The SLC images are available on request through the
  [ESA campaign portal](https://earth.esa.int/eogateway/campaigns/biosar-3). The P-band
  Pauli image was filtered with the
  [NL-SAR toolbox](https://www.charles-deledalle.fr/pages/nlsar.php) by Charles Deledalle.
- **Optical (Sweden)**: Lantmäteriet GSD-orthophoto 0.5 m RGB
  ([lantmateriet.se](https://www.lantmateriet.se)).
- **QB / WV**: QuickBird and WorldView images of San Francisco from the 2012 IEEE GRSS
  Data Fusion Contest.
- **Sentinel-1 / Sentinel-2**: Copernicus Sentinel data. `S1_*_GEE.tif` are temporal
  means of Sentinel-1 GRD images from Google Earth Engine.
- **Ku-band chips** (`*_sandiaKu*`): courtesy of Sandia National Laboratories, Radar ISR.

The Swedish images were post-processed at ONERA by Guillaume Brigot and Elise Koeniguer:
LIDAR and optical images were resampled onto the radar geometry from geocoding
information and SAR trajectories only, to provide a starting point for coregistration.
The GeFolki authors thank Malcolm Davidson and Michael Foumelis (ESA-ESRIN) for their
advice.

## Updating

`fetch.py` holds the commit, file list, sizes and hashes. To add a file, add its path,
size and `sha256sum` to `FILES`.
