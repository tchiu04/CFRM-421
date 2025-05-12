# Cont-Kukanov Order Allocation Backtesting

## Code Structure
- `backtest.py`: Main implementation containing:
  - Market data loading and processing
  - Cont-Kukanov allocator implementation
  - Execution simulation
  - Parameter search
  - Baseline strategies (Best Ask, TWAP, VWAP)
  - Results formatting and visualization

## Implementation Details
- **Allocator**: Implements the Cont-Kukanov algorithm with step size of 100 shares
- **Execution**: Simulates order execution across multiple venues
- **Parameter Search**: Random search over 100 combinations
- **Baseline Strategies**: Implements Best Ask, TWAP (60-second buckets), and VWAP

## Parameter Ranges
- `lambda_over`: (0.1, 2.0) - Penalty for overfilling
- `lambda_under`: (0.1, 2.0) - Penalty for underfilling
- `theta_queue`: (0.1, 1.0) - Queue risk penalty

These ranges were chosen to:
- Allow for both aggressive and conservative execution
- Balance between overfill and underfill penalties
- Maintain reasonable queue risk sensitivity

## Suggested Improvement: Queue Position Modeling
To improve fill realism, I suggest implementing a queue position model that:
1. Tracks the relative position of our orders in each venue's queue
2. Adjusts fill probabilities based on queue position
3. Incorporates queue position into the allocator's cost function

This would better reflect real-world execution where:
- Orders at the front of the queue have higher fill probabilities
- Queue position affects execution timing
- Market impact varies with queue position

## Usage
```bash
python backtest.py
```
Outputs:
- JSON results with execution statistics
- `results.png` showing cumulative costs