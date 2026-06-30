# Dataset files

Both datasets are stored as hourly grid tensors with shape
`(time, channel, width, height)`. The large `all_data.pkl` files are managed by
Git LFS.

## NYC

- Period: January--December 2013
- Shape: `(8760, 48, 20, 20)`
- Channels:
  - `0`: accident risk
  - `1:25`: hour-of-day one-hot features
  - `25:32`: day-of-week one-hot features
  - `32`: holiday indicator
  - `33:40`: POI features
  - `40`: temperature
  - `41:46`: clear, cloudy, rain, snow, and mist indicators
  - `46:48`: inflow and outflow

## Chicago

- Period: February--September 2016
- Shape: `(5832, 41, 20, 20)`
- Channels:
  - `0`: accident risk
  - `1:25`: hour-of-day one-hot features
  - `25:32`: day-of-week one-hot features
  - `32`: holiday indicator
  - `33`: temperature
  - `34:39`: clear, cloudy, rain, snow, and mist indicators
  - `39:41`: inflow and outflow

## Auxiliary files

- `risk_mask.pkl`: high-risk-region mask
- `risk_adj.pkl`: risk-similarity graph
- `road_adj.pkl`: road-similarity graph
- `poi_adj.pkl`: POI-similarity graph (NYC only)
- `grid_node_map.pkl`: mapping between graph nodes and grid cells

The files are preprocessed benchmark artifacts. Users are responsible for
confirming that their intended use and redistribution comply with the original
data-source terms.

SHA-256 checksums are listed in `SHA256SUMS`.
