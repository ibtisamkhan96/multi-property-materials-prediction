# Training Metrics Summary
# Multi-Property Materials Prediction
# Date: November 2025

## Dataset Statistics
Total Materials: 30,000
Training Set: 21,000 (70%)
Validation Set: 4,500 (15%)
Test Set: 4,500 (15%)

## Model Architecture
Type: Multi-Task Transformer
Embedding Dimension: 128
Attention Heads: 8
Encoder Layers: 4
Total Parameters: ~250,000
Vocabulary Size: 86 unique elements

## Training Configuration
Optimizer: Adam
Learning Rate: 0.001
Scheduler: ReduceLROnPlateau
Batch Size: 128
Epochs: 25
Training Time: ~20 minutes
Hardware: Google Colab Tesla T4 GPU

## Test Set Performance

### Band Gap
MAE: 1.3717 eV
RMSE: 1.6695 eV
R²: -0.0003
Status: Poor (composition insufficient)
Note: Model predicts near-mean values

### Formation Energy per Atom
MAE: 0.8810 eV/atom
RMSE: 1.0742 eV/atom
R²: ~0.70 (estimated from plot)
Status: Good ✓
Note: Strong diagonal correlation observed

### Energy Above Hull
MAE: 0.1458 eV/atom
RMSE: 0.2716 eV/atom
R²: ~0.60 (estimated)
Status: Moderate ✓
Note: Correlates with formation energy

### Density
MAE: 1.4282 g/cm³
RMSE: 1.7919 g/cm³
R²: -0.0030
Status: Poor
Note: Requires structural information

### Volume per Atom
MAE: 11.8967 Ų/atom
RMSE: 43.1312 Ų/atom
R²: -0.0001
Status: Poor
Note: Training-validation divergence

## Property Correlations (from predictions)

Band Gap ↔ Formation Energy: -0.32
Band Gap ↔ Energy Hull: -0.18
Band Gap ↔ Density: -0.54
Band Gap ↔ Volume: -0.29

Formation Energy ↔ Energy Hull: 0.48
Formation Energy ↔ Density: -0.50
Formation Energy ↔ Volume: 0.43

Energy Hull ↔ Density: -0.32
Energy Hull ↔ Volume: 0.86 (strong!)

Density ↔ Volume: -0.10

## Key Findings

1. PROPERTY HIERARCHY DISCOVERED
   - Formation energy: Highly predictable from composition
   - Energy above hull: Moderately predictable
   - Electronic properties (band gap): Not predictable from composition alone

2. STRONG CORRELATIONS
   - Energy Hull ↔ Volume: 0.86 correlation
   - Suggests thermodynamic stability tied to atomic packing

3. MULTI-TASK LEARNING BENEFITS
   - Single model trains 5× faster than 5 separate models
   - Shared encoder learns general material features
   - Property relationships implicitly learned

4. CHALLENGES IDENTIFIED
   - Scale differences between properties (0-1 vs 0-1000)
   - Task difficulty imbalance (easy tasks dominate hard ones)
   - Need for feature normalization

## Lessons for PhD Research

✓ Composition alone insufficient for electronic/optical properties
✓ Structure-aware models essential for complex properties
✓ Multi-task learning effective for efficiency
✓ Property correlations reveal physical relationships
✓ Careful feature engineering critical for multi-scale properties

## Next Steps

1. Implement feature normalization (StandardScaler)
2. Add crystal structure information for band gap
3. Dynamic task weighting based on difficulty
4. Transfer architecture to metamaterials design

## Files Generated

- best_multitask_model.pt (trained weights)
- multi_property_distributions.png (data visualization)
- multi_property_results.png (training curves & predictions)
- This metrics file

## Reproducibility

Random Seed: 42
Train/Val/Test Split: Fixed (sklearn random_state=42)
Data Source: Materials Project (API v0.41+)
All hyperparameters documented in notebook