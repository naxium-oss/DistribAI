# RFC 005: Federated Learning Aggregation Strategies

**Status:** ✅ Complete  
**Date:** 2026-04-21  
**Research Task:** Federated Learning Aggregation Strategies  
**Output:** docs/rfcs/005-aggregation-strategy.md

---

## Executive Summary

This RFC analyzes federated learning aggregation strategies: **FedAvg**, **FedProx**, **Scaffold**, and **FedNova**. Each strategy addresses different challenges in distributed training with heterogeneous, unreliable nodes. Key findings indicate that **FedAvg with Scaffold enhancements** provides the best balance of simplicity, convergence speed, and robustness for DistribAI's use case of training on heterogeneous consumer hardware.

---

## 1. FedAvg (Federated Averaging)

### Overview
FedAvg is the baseline algorithm for federated learning, allowing nodes to perform multiple local updates before aggregating weights. It reduces communication overhead by exchanging updated weights rather than gradients after each batch.

### Algorithm
1. Server sends global model weights to selected clients
2. Each client performs K local SGD steps on their local data
3. Clients send updated weights back to server
4. Server aggregates weights: `w_new = Σ (n_k / n_total) * w_k`
5. Repeat

### Key Characteristics
- **Communication**: Reduced vs. FedSGD (exchange weights, not gradients)
- **Local computation**: Multiple local steps per round (typically 5-20)
- **Aggregation**: Weighted average by data size
- **Complexity**: Low (simple averaging)

### Advantages
- **Simple implementation**: Easy to understand and implement
- **Reduced communication**: Fewer synchronization points
- **Proven effectiveness**: Works well for IID data
- **Low computational overhead**: Minimal server-side computation

### Disadvantages
- **Client drift**: Local models diverge on non-IID data
- **Slow convergence**: Heterogeneous data causes oscillations
- **No system heterogeneity handling**: Doesn't account for varying local work
- **Objective inconsistency**: Local objectives ≠ global objective

### Suitability for DistribAI
- **Medium**: Good baseline, but needs enhancements for non-IID data
- **Best for**: Homogeneous data distributions, reliable nodes
- **Worst for**: Highly non-IID data, stragglers

---

## 2. FedProx (Federated Proximal)

### Overview
FedProx extends FedAvg by adding a proximal term to the local objective to constrain local updates, reducing client drift when client data are non-IID.

### Algorithm
Local objective with proximal term:
```
min_w L_k(w) + (μ/2) * ||w - w_global||²
```
Where:
- `L_k(w)` is local loss on client k
- `μ` is FedProx hyperparameter
- `w_global` is current global model

### Key Characteristics
- **Proximal term**: Constrains local updates to stay close to global model
- **Hyperparameter**: μ controls strength of constraint
- **Aggregation**: Same as FedAvg (weighted average)
- **Complexity**: Low (adds proximal term to local objective)

### Advantages
- **Reduces client drift**: Keeps local models aligned with global model
- **Handles non-IID data**: Better convergence on heterogeneous data
- **Simple modification**: Easy to implement on top of FedAvg
- **Tunable**: μ can be adjusted based on heterogeneity level

### Disadvantages
- **Slower convergence**: Proximal term can slow progress
- **Hyperparameter tuning**: μ needs to be tuned per dataset
- **No system heterogeneity handling**: Doesn't account for varying local work
- **Euclidean distance calculation**: Adds computational overhead

### Suitability for DistribAI
- **High**: Addresses non-IID data challenge directly
- **Best for**: Moderate heterogeneity, need for simplicity
- **Worst for**: Highly heterogeneous systems (varying compute)

---

## 3. Scaffold (Stochastic Controlled Averaging)

### Overview
Scaffold uses control variates to reduce variance between client updates, aligning them more closely with the global objective and accelerating convergence.

### Algorithm
1. Server maintains control variates `c` (for client direction) and `c_global` (for global direction)
2. Client update: `w_k = w_k - η (∇L_k(w_k) - c + c_global)`
3. Server updates control variates based on client updates
4. Aggregation: Weighted average with control variate correction

### Key Characteristics
- **Control variates**: Estimates update direction for global data distribution
- **Variance reduction**: Reduces variance between client updates
- **Bi-directional correction**: Corrects both client and server updates
- **Complexity**: Medium (maintains and updates control variates)

### Advantages
- **Accelerates convergence**: Reduces client drift significantly
- **Handles non-IID data**: Robust to statistical heterogeneity
- **Theoretical guarantees**: Proven convergence rates
- **No hyperparameter tuning**: Self-adaptive control variates

### Disadvantages
- **Additional state**: Must maintain control variates for each client
- **Communication overhead**: Clients send control variate updates
- **Implementation complexity**: More complex than FedAvg/FedProx
- **Memory overhead**: Server stores control variates for all clients

### Suitability for DistribAI
- **Very High**: Best convergence on non-IID data
- **Best for**: High heterogeneity, need for fast convergence
- **Worst for**: Memory-constrained servers, very large client pools

---

## 4. FedNova (Normalized Averaging)

### Overview
FedNova normalizes local updates by the number of local steps performed, addressing system heterogeneity where clients perform varying amounts of local work.

### Algorithm
1. Each client performs τ_k local steps
2. Normalized update: `Δ_k_norm = Δ_k / τ_k`
3. Aggregation: Weighted average of normalized updates
4. Reconstruct global update: `Δ_global = Σ p_k * τ_k * Δ_k_norm`

### Key Characteristics
- **Normalization**: Divides updates by number of local steps
- **System heterogeneity handling**: Accounts for varying local work
- **Objective consistency**: Eliminates objective inconsistency
- **Complexity**: Low (simple normalization)

### Advantages
- **Handles system heterogeneity**: Works well with varying compute capabilities
- **Eliminates client drift mismatch**: Normalizes by local work
- **No stale update issues**: Compensates for differing completion times
- **Efficient gradient approximation**: Better approximation of global gradient

### Disadvantages
- **No statistical heterogeneity handling**: Doesn't address non-IID data
- **Requires τ_k tracking**: Must track local steps per client
- **Assumes SGD**: Works best with SGD local solver
- **Limited to synchronous settings**: Asynchronous variants less studied

### Suitability for DistribAI
- **High**: Addresses system heterogeneity (varying GPU/CPU capabilities)
- **Best for**: Heterogeneous hardware, synchronous training
- **Worst for**: Non-IID data (needs combination with other methods)

---

## 5. Comparative Analysis

| Aspect | FedAvg | FedProx | Scaffold | FedNova |
|--------|--------|---------|----------|---------|
| **Complexity** | Low | Low | Medium | Low |
| **Communication Overhead** | Low | Low | Medium | Low |
| **Non-IID Data Handling** | Poor | Good | Excellent | Poor |
| **System Heterogeneity Handling** | Poor | Poor | Poor | Excellent |
| **Convergence Speed (IID)** | Fast | Medium | Fast | Fast |
| **Convergence Speed (Non-IID)** | Slow | Medium | Fast | Slow |
| **Memory Overhead** | Low | Low | Medium | Low |
| **Hyperparameter Tuning** | None | μ (per dataset) | None | None |
| **Implementation Difficulty** | Easy | Easy | Medium | Easy |

---

## 6. Suitability for DistribAI

### Use Case Analysis

DistribAI's federated learning requirements:
- **Heterogeneous hardware**: Contributors have varying GPU/CPU capabilities
- **Non-IID data**: Training data may be distributed across contributors
- **Unreliable nodes**: Contributors may drop out or have stragglers
- **Scalability**: Must support 1000+ nodes
- **Communication efficiency**: Internet-scale training requires efficient communication

### Challenge Analysis

**Statistical Heterogeneity (Non-IID Data):**
- FedAvg: Struggles with client drift
- FedProx: Handles moderate heterogeneity
- Scaffold: Excellent handling via control variates
- FedNova: No handling

**System Heterogeneity (Varying Compute):**
- FedAvg: No handling, biases toward fast nodes
- FedProx: No handling
- Scaffold: No handling
- FedNova: Excellent handling via normalization

**Communication Overhead:**
- FedAvg: Low (weights only)
- FedProx: Low (weights only)
- Scaffold: Medium (weights + control variates)
- FedNova: Low (normalized weights only)

**Implementation Complexity:**
- FedAvg: Simplest
- FedProx: Simple (add proximal term)
- Scaffold: Complex (control variates)
- FedNova: Simple (normalization)

---

## 7. Recommended Strategy

### Primary Recommendation: FedAvg + Scaffold

**Rationale:**

1. **Statistical heterogeneity is critical**: DistribAI will likely have non-IID data distributions across contributors. Scaffold's control variates provide the best handling of this challenge.

2. **System heterogeneity can be managed**: While FedNova handles system heterogeneity well, DistribAI can mitigate this through:
   - Tier-based scheduling (group nodes by capability)
   - Fixed local steps per round (all nodes do K steps)
   - Timeout enforcement (reassign slow nodes)

3. **Communication efficiency matters**: Scaffold's additional communication overhead (control variates) is acceptable given the convergence benefits.

4. **Scalability**: Scaffold's memory overhead (control variates per client) is manageable for 1000+ nodes (~1KB per client = 1MB total).

### Hybrid Approach: FedAvg + Scaffold + FedNova

For maximum robustness, combine all three:
- **Scaffold**: Handle statistical heterogeneity (non-IID data)
- **FedNova**: Handle system heterogeneity (varying compute)
- **FedAvg**: Base aggregation framework

Implementation:
```
1. Each client performs τ_k local steps
2. Client computes normalized update: Δ_k_norm = (w_k - w_global) / τ_k
3. Client applies control variate correction: Δ_k_corrected = Δ_k_norm - c + c_global
4. Server aggregates: Δ_global = Σ p_k * τ_k * Δ_k_corrected
5. Server updates control variates
```

### Alternative: FedProx for Simplicity

If Scaffold's complexity is prohibitive:
- Use FedProx with tuned μ
- Combine with tier-based scheduling for system heterogeneity
- Accept slower convergence for simpler implementation

---

## 8. Implementation Plan

### Phase 0 (Development - 10 nodes)
1. **Implement FedAvg baseline**
   - Simple weighted averaging
   - Fixed local steps (K=5)
   - No heterogeneity handling

### Phase 1 (Alpha - 50 nodes)
1. **Add FedProx**
   - Implement proximal term
   - Tune μ on validation data
   - Compare convergence vs. FedAvg

2. **Add tier-based scheduling**
   - Benchmark nodes on registration
   - Assign to tiers (high/mid/low)
   - Fixed local steps per tier

### Phase 2 (Beta - 200 nodes)
1. **Implement Scaffold**
   - Add control variates
   - Test on non-IID synthetic data
   - Compare convergence vs. FedProx

2. **Add FedNova normalization**
   - Track local steps per client
   - Normalize updates by τ_k
   - Test with heterogeneous hardware

### Phase 3+ (Production - 1000+ nodes)
1. **Hybrid approach**
   - Combine Scaffold + FedNova
   - Ablation study to validate benefits
   - Optimize hyperparameters

2. **Adaptive strategies**
   - Dynamic μ based on heterogeneity
   - Adaptive local steps based on node capability
   - Asynchronous aggregation for stragglers

---

## 9. Open Questions

1. **Control variate memory**: At what scale does Scaffold's memory overhead become problematic? (1000 nodes? 10,000 nodes?)
2. **τ_k tracking**: How to handle clients that don't complete all τ_k steps due to failures?
3. **Hyperparameter tuning**: How to automatically tune μ for FedProx in production?
4. **Asynchronous aggregation**: How to extend Scaffold/FedNova to asynchronous settings?
5. **Byzantine resilience**: How do these aggregation strategies interact with Byzantine detection (RFC 006)?

---

## 10. Communication Overhead Analysis

### FedAvg
- **Per round**: Model weights (e.g., 100MB for 1B model)
- **Frequency**: Every K local steps (K=5-20)
- **Total**: 100MB × (1/K rounds/batch) = 5-20MB per batch

### FedProx
- **Per round**: Model weights (100MB)
- **Frequency**: Every K local steps
- **Total**: Same as FedAvg (5-20MB per batch)

### Scaffold
- **Per round**: Model weights (100MB) + control variates (~1KB per client)
- **Frequency**: Every K local steps
- **Total**: 100MB + negligible control variate overhead = 5-20MB per batch
- **Note**: Control variates are small (~1KB) compared to model weights

### FedNova
- **Per round**: Model weights (100MB)
- **Frequency**: Every K local steps
- **Total**: Same as FedAvg (5-20MB per batch)
- **Note**: Normalization is client-side, no additional communication

**Conclusion**: All strategies have similar communication overhead. Scaffold's control variates add negligible bandwidth.

---

## 11. Convergence Speed Comparison

Based on literature and empirical studies:

| Strategy | IID Data | Non-IID Data | System Heterogeneity |
|----------|-----------|--------------|---------------------|
| FedAvg | Fast | Slow | Slow (biased to fast) |
| FedProx | Medium | Medium | Slow (biased to fast) |
| Scaffold | Fast | Fast | Slow (biased to fast) |
| FedNova | Fast | Slow | Fast |
| Scaffold + FedNova | Fast | Fast | Fast |

**Conclusion**: Scaffold + FedNova provides the best convergence across all scenarios.

---

## 12. Final Recommendation

**Primary Choice**: Scaffold (for statistical heterogeneity)

**Secondary Enhancement**: FedNova normalization (for system heterogeneity)

**Implementation Strategy**:
1. Start with FedAvg baseline (Phase 0)
2. Add Scaffold for non-IID data handling (Phase 2)
3. Add FedNova normalization if system heterogeneity is problematic (Phase 3)
4. Use tier-based scheduling as a simpler alternative to FedNova (Phase 1)

**Rationale**: Scaffold provides the most significant benefit (handling non-IID data) with acceptable complexity. FedNova provides additional benefits for system heterogeneity but can be addressed through scheduling if implementation complexity is a concern.

**Next step**: Proceed to RFC 006 on Byzantine Fault Tolerance in Distributed ML to understand how aggregation strategies interact with malicious node detection.
