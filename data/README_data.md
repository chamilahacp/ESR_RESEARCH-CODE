# Data description

This project uses a tabular dataset containing the final predictor variables and response variable used in the machine-learning models.

## Response variable
- `A` : Air Pollution Tolerance Index (APTI)

## Predictor variables
- `N6` : NO₂ exposure variable - 6 month exposure 
- `S4` : SO₂ exposure variable - 2 month exposure 
- `T4` : temperature variable - 2 month exposure 
- `P5` : PM₂.₅-AOD drived variable - 3 month exposure 
- `O5` : O₃ exposure variable month exposure 
- `R4` : rainfall variablemonth exposure 
- `H5` : relative humidity variable month exposure 
- `S`  : deciduous-state indicator - only for deciduous

## Notes
- Missing values were removed before analysis.
- Predictor and response columns were converted to numeric values prior to model fitting.
- If the raw dataset is not publicly shared, please provide access upon reasonable request or replace with a processed version if permitted.
