# Data Directory

External datasets used by the GBC book experiments. Place CSV files here before running the reproduction scripts in `scripts/`.

## Datasets

### Phantom & Star (Ch 8)

Export from R's `jumpgp` package:

```r
# install.packages("remotes")
# remotes::install_bitbucket("gramacylab/jumpgp")
library(jumpgp)
data(phantom); write.csv(phantom, "phantom.csv", row.names = FALSE)
data(star);    write.csv(star,    "star.csv",    row.names = FALSE)
```

### LGBB Rocket (Ch 9)

Download `lgbb_fill.csv` from Bobby Gramacy's surrogates site:
<https://bobby.gramacy.com/surrogates/>

3 inputs (mach_s, alpha_s, beta_s) → 1 response (side_c). ~37K rows.

### Lake Temperature (Ch 14)

From Holthuijzen et al. (2025), "Synthesizing data products, mathematical models, and observational measurements for lake temperature forecasting."

281K rows. Features: DOY, Depth, Horizon, phi, GLM_mean, GLM_std → target: temp_obs.

Contact the authors or see the paper's data availability statement for access.
