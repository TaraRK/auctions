# Difference Between `max_bid` and `max_payment`

## Definitions

### `max_bid`
- **What it is**: The highest bid submitted by any agent
- **Formula**: `max_bid = max(b_1, b_2, ..., b_n)` where `b_i` are the bids
- **Example**: If agents bid `[0.3, 0.5, 0.7, 0.4]`, then `max_bid = 0.7`

### `max_payment`
- **What it is**: The maximum payment the auctioneer is allowed to charge
- **Formula**: `max_payment = max_bid × 1.5`
- **Example**: If `max_bid = 0.7`, then `max_payment = 0.7 × 1.5 = 1.05`

## Key Differences

| Aspect | `max_bid` | `max_payment` |
|--------|-----------|--------------|
| **What it represents** | Highest bid from agents | Maximum allowed payment |
| **Who determines it** | Agents (through their bids) | Auctioneer (constraint) |
| **Relationship** | `max_payment = max_bid × 1.5` | Always 1.5× larger than `max_bid` |
| **Purpose** | Shows what agents are willing to pay | Limits what auctioneer can charge |

## Code Location

```python
# Line 142-144: Calculate max_bid (highest bid)
if bids.dim() == 1:
    max_bid = torch.max(bids).item()  # <-- max_bid: highest bid from agents
else:
    max_bid = torch.max(bids, dim=-1)[0].item()

# Line 145: Calculate max_payment (payment constraint)
max_payment = max_bid * 1.5  # <-- max_payment: 1.5× max_bid (payment limit)

# Line 146: Clamp payments to this limit
payments = torch.clamp(payments, min=0.0, max=max_payment)
```

## Why This Matters

### In Standard Auctions
- **First-price**: Payment = winning bid (payment ≤ max_bid)
- **Second-price**: Payment = second-highest bid (payment ≤ max_bid)

### In This AMD Framework
- **Payment can exceed max_bid**: The auctioneer can charge up to `1.5 × max_bid`
- **Example**: 
  - If `max_bid = 0.8`, then `max_payment = 1.2`
  - The auctioneer could charge `1.2` even though no one bid that high
  - This allows revenue to exceed 1.0 when bids are high

## Implications

1. **Revenue can exceed bids**: Since `max_payment = 1.5 × max_bid`, revenue can be higher than any bid
2. **Individual rationality violation**: If payment > value, agent gets negative utility
3. **Learning flexibility**: The 1.5× multiplier gives the payment network room to learn optimal pricing

## Example Scenario

```
Agents bid: [0.2, 0.5, 0.8, 0.3]
max_bid = 0.8  (highest bid)
max_payment = 0.8 × 1.5 = 1.2  (maximum allowed payment)

Possible outcomes:
- Payment network could charge: 0.0 to 1.2
- In first-price: would charge 0.8 (the winning bid)
- In this AMD: could charge up to 1.2 (1.5× the max bid)
```

