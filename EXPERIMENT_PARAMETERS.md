# Experiment Parameters Guide

This document outlines reasonable parameter ranges for AMD experiments and what to expect from different combinations.

## Parameters to Vary

### 1. Learning Rate (`lr_auctioneer`)
**What it controls:** How fast the auctioneer learns to adjust payments and allocations

**Recommended values:**
- `1e-4` (very conservative, slow learning)
- `2e-4` (moderate, current default)
- `5e-4` (faster learning, may be unstable)
- `1e-3` (aggressive, likely to cause revenue explosion)

**Expected effects:**
- **Low (1e-4)**: Slower convergence, more stable, less revenue explosion
- **High (5e-4+)**: Faster learning, but may overshoot and cause instability

**Recommendation:** Start with `[1e-4, 2e-4, 5e-4]`

---

### 2. Update Epochs (`update_epochs`)
**What it controls:** How many times PPO updates on the same batch of experiences

**Recommended values:**
- `2` (minimal updates, faster)
- `4` (standard, current default)
- `8` (more thorough learning, slower)

**Expected effects:**
- **Low (2)**: Faster training, may underfit
- **High (8)**: Better learning from each batch, but slower and may overfit

**Recommendation:** Start with `[2, 4, 8]`

---

### 3. Gamma (Discount Factor)
**What it controls:** How much agents/auctioneer value future rewards vs immediate rewards

**Recommended values:**
- `0.0` (myopic, only immediate rewards matter)
- `0.5` (moderate future consideration)
- `0.99` (heavily weights future, enables collusion)

**Expected effects:**
- **Gamma = 0.0**: No collusion possible, agents optimize for each auction independently
- **Gamma = 0.5**: Some future consideration, limited coordination
- **Gamma = 0.99**: Enables collusion/coordination, agents learn to cooperate

**Recommendation:** Start with `[0.0, 0.5, 0.99]` to test collusion hypothesis

---

### 4. Information Type (`info_type`)
**What it controls:** What information is revealed to agents after each auction

**Available types:**
- `MINIMAL`: Only win/loss, own payment
- `WINNER`: Winner's bid revealed
- `LOSER`: Losing bids revealed (for winner)
- `FULL_TRANSPARENCY`: All bids revealed
- `FULL_REVELATION`: All bids, values, and payments revealed

**Expected effects:**
- **MINIMAL**: Agents have least information, harder to coordinate
- **FULL_REVELATION**: Agents have most information, easier to coordinate/collude

**Recommendation:** Start with `[MINIMAL, FULL_REVELATION]` to test information impact

---

### 5. Training Interval
**What it controls:** How long each phase lasts (agents learn vs auctioneer learns)

**Recommended values:**
- `100` (short phases, frequent switching)
- `200` (moderate, current default)
- `500` (long phases, more stable but slower feedback)

**Expected effects:**
- **Short (100)**: Faster feedback, but may be unstable
- **Long (500)**: More stable, better feedback loop observation

**Recommendation:** Keep at `200` for now, or test `[100, 200, 500]`

---

## Recommended Experiment Sets

### Set 1: Basic Parameter Sweep (27 experiments)
```python
learning_rates = [1e-4, 2e-4, 5e-4]
update_epochs_list = [2, 4, 8]
gammas = [0.0, 0.5, 0.99]
info_types = [InformationType.FULL_REVELATION]  # Keep fixed for now
training_interval = 200  # Keep fixed
```
**Total: 3 × 3 × 3 = 27 experiments**

**What to look for:**
- Which gamma values enable collusion (lower theta convergence)?
- How does learning rate affect stability?
- Optimal update epochs for convergence speed?

---

### Set 2: Information Impact (18 experiments)
```python
learning_rates = [2e-4]  # Keep fixed
update_epochs_list = [4]  # Keep fixed
gammas = [0.0, 0.5, 0.99]
info_types = [InformationType.MINIMAL, InformationType.FULL_REVELATION]
training_interval = 200
```
**Total: 1 × 1 × 3 × 2 = 6 experiments**

**What to look for:**
- Does information revelation enable collusion?
- How does information affect revenue?

---

### Set 3: Collusion Test (9 experiments)
```python
learning_rates = [2e-4]  # Keep fixed
update_epochs_list = [4]  # Keep fixed
gammas = [0.0, 0.5, 0.99]  # Vary this to test collusion
info_types = [InformationType.FULL_REVELATION]  # Full info to enable collusion
training_interval = 200
```
**Total: 3 experiments**

**What to look for:**
- **Gamma = 0.0**: Should see high theta (aggressive bidding), no collusion
- **Gamma = 0.99**: Should see lower theta (coordination), possible collusion
- Compare revenue and efficiency across gamma values

---

### Set 4: Full Sweep (54 experiments)
```python
learning_rates = [1e-4, 2e-4, 5e-4]
update_epochs_list = [2, 4, 8]
gammas = [0.0, 0.5, 0.99]
info_types = [InformationType.MINIMAL, InformationType.FULL_REVELATION]
training_interval = 200
```
**Total: 3 × 3 × 3 × 2 = 54 experiments**

**What to look for:**
- Optimal parameter combination
- Interaction effects between parameters
- Best revenue/efficiency trade-off

---

## Expected Results by Parameter

### High Gamma (0.99) + Full Revelation
- **Expected:** Lower theta convergence (agents coordinate)
- **Revenue:** May be lower (agents bid lower together)
- **Efficiency:** Should remain high (highest value still wins)
- **Interpretation:** Collusion/coordination enabled

### Low Gamma (0.0) + Any Info
- **Expected:** Higher theta (agents bid aggressively)
- **Revenue:** Higher (agents compete more)
- **Efficiency:** Should remain high
- **Interpretation:** No collusion, competitive equilibrium

### High Learning Rate (5e-4) + High Gamma
- **Expected:** May cause instability or revenue explosion
- **Risk:** System may not converge
- **Interpretation:** Too aggressive learning

### Low Learning Rate (1e-4) + Any Gamma
- **Expected:** Slower convergence, more stable
- **Revenue:** More stable over time
- **Interpretation:** Conservative but reliable

---

## Success Criteria

### Convergence Indicators:
1. **Revenue stability:** Std < 0.05 in last 1000 rounds
2. **Theta stability:** Std < 0.05 in last 1000 rounds
3. **Change < 5%:** Last 1k vs previous 1k rounds
4. **Visual:** Flat lines in convergence plots

### Optimal Parameters Should:
1. **Converge** (stable revenue/theta)
2. **High efficiency** (close to 1.0)
3. **Reasonable revenue** (0.15-0.30 for values in [0,1])
4. **No explosion** (revenue < 1.0)

---

## Running Experiments

### Quick Test (6 experiments):
```python
learning_rates = [2e-4]
update_epochs_list = [4]
gammas = [0.0, 0.99]
info_types = [InformationType.MINIMAL, InformationType.FULL_REVELATION]
```

### Medium Test (27 experiments):
```python
learning_rates = [1e-4, 2e-4, 5e-4]
update_epochs_list = [2, 4, 8]
gammas = [0.0, 0.5, 0.99]
info_types = [InformationType.FULL_REVELATION]
```

### Full Test (54 experiments):
Use Set 4 above (will take longer but comprehensive)

---

## Analysis After Experiments

1. **Find best result:** Highest revenue with stable convergence
2. **Compare gamma effects:** Does 0.99 enable collusion?
3. **Information impact:** Does FULL_REVELATION help?
4. **Learning rate sweet spot:** Balance between speed and stability
5. **Check P&G outputs:** Are payments reasonable? Who wins most?

---

## Notes

- **N_rounds:** Keep at 30000 for convergence
- **N_agents:** Keep at 10 (standard)
- **Training interval:** 200 is a good default
- **Buffer size:** Automatically matches training_interval

Each experiment will generate:
- Convergence plots
- Strategy visualizations  
- P&G network outputs
- JSON logs with all metrics
- Summary identifying best result

