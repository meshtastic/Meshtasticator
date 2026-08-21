# Batumi Radio Calibration

This report documents the reusable radio calibration used by the packaged
`batumi` preset. It is intentionally aggregate-only: no node names, source node
IDs, or collection endpoint details are required for reproducing simulator
behavior.

## Data Window

The reference data is a 30-day Batumi/Georgia-area neighbor-SNR snapshot. Nodes
were filtered to the preset bounding box:

```text
lat: 41.50..41.82
lon: 41.50..41.86
```

The land-cover clutter grid is derived from public OpenStreetMap building,
landuse, natural, and water polygons fetched with Overpass. The packaged CSV is
a coarse 500 m raster with only `open`, `urban`, `water`, and `forest` classes;
it does not include raw OSM feature IDs or names. Attribution: OpenStreetMap
contributors.

Reference sample shape:

```text
nodes in bbox:                      92
current neighbor edges:             85
30-day distinct directed edges:     296
30-day neighbor samples:            14361
OSM clutter cells:                  4320
OSM clutter cells by class:         urban=1209 open=3101 water=5 forest=5
```

Observed median SNR across the 296 directed calibration edges:

```text
min:  -20.75 dB
p05:  -19.06 dB
p25:  -17.50 dB
p50:  -10.88 dB
mean:  -9.53 dB
p75:   -3.75 dB
p95:    5.81 dB
max:    6.75 dB
```

## Calibration Shape

The calibration is not a node-pair replay table. The runtime simulator never
asks "was this exact directed pair observed?" and never lifts one specific link
because it appeared in the calibration data.

Instead, the preset stores a reusable feature transform:

```text
calibrated_snr = intercept
               + raw_snr_clip * a
               + log_distance_km * b
               + log_distance_km_sq * c
               + terrain/clutter/vantage/land-cover feature terms
```

The model is trained from two kinds of examples:

```text
positive targets: 296 observed directed links with median observed SNR
background targets: all other generated directed pairs, weakly weighted at 0.02
background target SNR: min(raw_model_snr, -22 dB)
ridge lambda: 50
```

The background targets are deliberately weak evidence. A missing 30-day neighbor
edge does not prove the link is impossible, but it is enough to stop a
positive-only fit from making the whole city reachable.

The applied coefficient set lives in `presets/batumi.yaml` under
`radio_calibration.link_calibration_model.coefficients`. Runtime packet logic
uses only those coefficients and path features, so the same model can be applied
to new generated points that have no ground-truth links.

## Scalar Baseline

The scalar baseline includes the preset noise floor, path-loss distance floor,
terrain, and OSM-derived clutter, but not the fitted feature transform:

```yaml
radio_calibration:
  noise_level: -110.5
  path_loss_distance_floor_m: 780.0
  reported_snr_min_db: -21.25
  reported_snr_max_db: 8.25
```

On the 296 observed directed edges, scalar-only reachability is poor:

```text
observed directed links reachable by scalar model: 25 / 296
scalar-only sensed directed links across all generated pairs: 650 / 8372
```

Scalar model RSSI margin to modem sensitivity on observed pairs:

```text
min: -100.33 dB
p05:  -87.71 dB
p25:  -62.65 dB
p50:  -47.17 dB
mean: -43.76 dB
p75:  -25.39 dB
p95:    5.99 dB
max:   22.08 dB
```

Uncapped scalar model SNR for the observed pairs:

```text
min: -121.33 dB
p05: -108.71 dB
p25:  -83.65 dB
p50:  -68.17 dB
mean: -64.76 dB
p75:  -46.39 dB
p95:  -15.01 dB
max:    1.08 dB
```

Pairwise residual, defined here as `observed_median_snr - scalar_model_snr`, is
large because coarse map pins, balcony/roof placement, antenna orientation,
coastal corridors, and reflections are not fully represented by distance,
terrain, and OSM land-cover alone:

```text
min:  -19.83 dB
p05:    4.72 dB
p25:   34.57 dB
p50:   55.77 dB
mean:  55.23 dB
p75:   79.48 dB
p95:  103.34 dB
max:  118.83 dB
```

OSM clutter loss on the same observed directed links:

```text
min:   0.00 dB
p05:   3.61 dB
p25:   7.32 dB
p50:  15.01 dB
mean: 14.42 dB
p75:  22.21 dB
p95:  25.00 dB
max:  25.00 dB
```

## Fitted Feature Model

After applying the reusable feature transform:

```text
observed directed links reachable by fitted model: 88 / 296
fitted-model sensed directed links across all generated pairs: 1704 / 8372
```

This does not force every observed calibration edge to be reachable. That is
intentional: if a link needs pair-specific information to exist, the generic
model treats it as uncertainty instead of baking it into runtime physics.

Reported model SNR for the 296 observed calibration links:

```text
min: -21.25 dB
p05: -21.25 dB
p25: -21.25 dB
p50: -21.25 dB
mean: -19.70 dB
p75: -19.99 dB
p95: -13.09 dB
max:  -1.57 dB
```

Residual after feature calibration, `observed_median_snr - fitted_model_snr`:

```text
min: -15.00 dB
p05:  -0.56 dB
p25:   3.50 dB
p50:   8.85 dB
mean: 10.17 dB
p75:  16.26 dB
p95:  26.59 dB
max:  28.00 dB
```

Reported SNR distribution across all generated directed pairs after feature
calibration:

```text
min: -21.25 dB
p05: -21.25 dB
p25: -21.25 dB
p50: -21.25 dB
mean: -20.05 dB
p75: -21.25 dB
p95: -12.15 dB
max:   8.25 dB
```

## Why Not Pairwise Correction

The runtime model deliberately does not carry a lookup table of observed
directed links and does not boost one exact node pair just because that pair
appeared in the calibration sample. A pair-specific correction can make the
calibration set look perfect while adding nothing for a new generated point
with no ground truth.

The packaged observations are training/evaluation records only. The simulator
applies one fitted transform to every generated TX/RX pair. That is less
flattering to the calibration set, but much more useful for testing new
placements and other nearby meshes.

## Known Limitations

This calibration target is neighbor-SNR history, not packet-level PER trace.
Neighbor tables are biased toward nodes that report neighbor info, and a 30-day
observed edge does not prove the link is continuously available.

The fitted coefficients are local to the packaged Batumi preset. They do not
change random/default simulations and should not be treated as universal LoRa
propagation constants. The useful part to reuse elsewhere is the workflow:
compute physical path features, fit coefficients against local observations,
and evaluate generated pairs without runtime per-link priors.
