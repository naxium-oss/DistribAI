# RFC 006: Byzantine Fault Tolerance in Distributed ML

**Status:** ✅ Complete  
**Date:** 2026-04-21  
**Research Task:** Byzantine Fault Tolerance in Distributed ML  
**Output:** docs/rfcs/006-byzantine-fault-tolerance.md

---

## Executive Summary

This RFC analyzes Byzantine fault tolerance methods for distributed machine learning, focusing on **Coordinate-wise Median**, **Trimmed Mean**, **Krum/Multi-Krum**, and **AUROR** (clustering-based detection). These methods detect and mitigate malicious nodes that send corrupted or adversarial updates. Key findings indicate that **Multi-Krum combined with Trimmed Mean** provides the best balance of robustness, computational efficiency, and convergence speed for DistribAI's use case.

---

## 1. Problem Statement

### Byzantine Faults in Distributed ML

In distributed training, **Byzantine faults** occur when nodes behave maliciously or arbitrarily, sending corrupted gradients or model updates. Unlike crash failures, Byzantine nodes can:
- Send arbitrary gradient values (e.g., large magnitudes, wrong directions)
- Collude with other malicious nodes
- Adapt their behavior based on system state
- Exploit aggregation algorithms to poison the global model

### Threat Model

**Assumptions:**
- Up to `f` Byzantine nodes out of `n` total nodes
- Byzantine nodes can send arbitrary updates
- Byzantine nodes may collude
- Byzantine nodes know the aggregation algorithm
- Honest nodes follow the training protocol correctly

**Goal:** Ensure convergence to the correct model despite up to `f` Byzantine nodes.

---

## 2. Coordinate-wise Median

### Overview
Coordinate-wise median replaces the mean with the median for each dimension of the gradient vector independently.

### Algorithm
For `n` update vectors `v_1, v_2, ..., v_n`, each of dimension `d`:
```
For each dimension j = 1 to d:
    v_agg[j] = median(v_1[j], v_2[j], ..., v_n[j])
```

### Key Characteristics
- **Computation**: O(n log n) per dimension (for sorting)
- **Memory**: O(n) per dimension
- **Robustness**: Tolerates up to ~50% Byzantine nodes (under certain assumptions)
- **Complexity**: Low (simple median calculation)

### Advantages
- **Computationally efficient**: Faster than geometric median
- **Simple implementation**: Easy to understand and implement
- **Strong robustness**: Can tolerate nearly 50% Byzantine nodes
- **Parallelizable**: Each dimension computed independently

### Disadvantages
- **Independent dimensions**: Treats each dimension independently, missing correlated attacks across dimensions
- **Weaker theoretical guarantees**: Less strong than geometric median
- **Sensitive to outliers**: Can be affected if Byzantine nodes coordinate their attacks per dimension

### Suitability for DistribAI
- **Medium**: Good for simple attacks, but vulnerable to sophisticated multi-dimensional attacks
- **Best for**: Low-complexity environments with simple threat models
- **Worst for**: Sophisticated adversarial attacks with coordinated multi-dimensional updates

---

## 3. Trimmed Mean

### Overview
Trimmed mean discards a fraction of the most extreme values (smallest and largest) for each dimension before averaging the remaining values.

### Algorithm
For each dimension `j`:
1. Sort values: `v_1[j], v_2[j], ..., v_n[j]`
2. Remove `βn` smallest and `βn` largest values (where `β` is the trimming fraction)
3. Compute mean of remaining `(1-2β)n` values:
```
v_agg[j] = mean(remaining values)
```

### Key Characteristics
- **Computation**: O(n log n) per dimension (for sorting)
- **Memory**: O(n) per dimension
- **Robustness**: Tolerates up to `βn` Byzantine nodes
- **Complexity**: Low (sorting + mean)

### Advantages
- **Computationally reasonable**: Efficient sorting-based approach
- **Tunable**: `β` can be adjusted based on expected Byzantine fraction
- **Better than mean**: More robust than simple averaging
- **Simple to implement**: Straightforward algorithm

### Disadvantages
- **Sensitive to `β`**: 
  - Too low: Vulnerable to Byzantine influence
  - Too high: Discards honest updates, slows convergence
- **Non-IID sensitivity**: May discard legitimate updates in heterogeneous (non-IID) settings
- **Independent dimensions**: Like median, treats dimensions independently
- **Requires accurate estimate**: Must know or estimate Byzantine fraction

### Suitability for DistribAI
- **Medium**: Good if Byzantine fraction is known and stable
- **Best for**: Environments with known or estimated Byzantine fraction
- **Worst for**: Highly non-IID data or unknown Byzantine fraction

---

## 4. Krum and Multi-Krum

### Overview
Krum selects the single update most "representative" or "central" based on distance to nearest neighbors. Multi-Krum averages the `m` most central updates.

### Algorithm

**Krum:**
For each update `v_i`:
1. Find `k` nearest neighbors (excluding `v_i` itself)
2. Compute score: `score(i) = Σ_{l ∈ N_k(i)} ||v_i - v_l||²`
3. Select update with minimum score: `i* = argmin_i score(i)`
4. Aggregated update: `v_agg = v_i*`

**Multi-Krum:**
1. Compute Krum scores for all updates
2. Select `m` updates with lowest scores
3. Aggregate: `v_agg = mean(selected updates)`

**Parameter selection:**
- `k = n - f - 2` (ensures neighborhood excludes all Byzantine nodes)
- `f` = maximum number of Byzantine nodes
- `m` = typically 5-10 (small number of central updates)

### Key Characteristics
- **Computation**: O(n²) for pairwise distances (expensive for large `n`)
- **Memory**: O(n²) for distance matrix
- **Robustness**: Tolerates up to `(n-k-1)/2` Byzantine nodes
- **Complexity**: Medium (pairwise distance calculations)

### Advantages
- **Global consideration**: Considers entire update vector, not just dimensions independently
- **Robust to correlated attacks**: Multi-dimensional attacks harder to evade
- **Provable guarantees**: Strong theoretical robustness guarantees
- **Multi-Krum stability**: Averaging multiple central updates improves stability

### Disadvantages
- **Computationally expensive**: O(n²) pairwise distance calculations
- **Memory intensive**: Distance matrix scales quadratically
- **Slow convergence with single selection**: Krum (single update) can be unstable
- **Parameter sensitivity**: Requires accurate estimate of `f` (max Byzantine nodes)

### Suitability for DistribAI
- **High**: Best robustness against sophisticated attacks
- **Best for**: Medium-scale deployments (n < 1000) with sophisticated threat models
- **Worst for**: Very large deployments (n > 10,000) due to O(n²) complexity

---

## 5. AUROR (Clustering-Based Detection)

### Overview
AUROR uses clustering to partition updates into two classes: malicious and benign. The smaller class is identified as malicious and filtered out.

### Algorithm
1. Collect all client updates
2. Cluster updates into two groups using clustering algorithm
3. Identify smaller cluster as malicious class
4. Filter out updates from malicious cluster
5. Aggregate remaining updates (e.g., using mean)

### Key Characteristics
- **Computation**: Depends on clustering algorithm (typically O(n²) or O(n log n))
- **Memory**: O(n²) for distance-based clustering
- **Robustness**: Assumes malicious nodes form minority cluster
- **Complexity**: Medium (clustering + filtering)

### Advantages
- **Unsupervised**: No need to know Byzantine fraction
- **Adaptive**: Can adapt to changing attack patterns
- **Multi-dimensional**: Considers entire update vector
- **Flexible**: Can use various clustering algorithms

### Disadvantages
- **Assumption dependency**: Assumes malicious nodes form distinct cluster
- **Computationally expensive**: Clustering can be expensive
- **Cluster quality sensitive**: Poor clustering leads to poor detection
- **Scalability issues**: Clustering scales poorly with large `n`

### Suitability for DistribAI
- **Medium**: Good when Byzantine fraction is unknown
- **Best for**: Environments with unknown or varying Byzantine fraction
- **Worst for**: Very large deployments or when malicious nodes don't form distinct clusters

---

## 6. Comparative Analysis

| Aspect | Coordinate-wise Median | Trimmed Mean | Krum | Multi-Krum | AUROR |
|--------|------------------------|--------------|------|------------|-------|
| **Computation** | O(n log n) | O(n log n) | O(n²) | O(n²) | O(n²) |
| **Memory** | O(n) | O(n) | O(n²) | O(n²) | O(n²) |
| **Robustness** | ~50% | β-dependent | (n-k-1)/2 | (n-k-1)/2 | Minority assumption |
| **Convergence Speed** | Medium | Medium | Slow (single) | Fast | Medium |
| **Multi-dimensional** | No | No | Yes | Yes | Yes |
| **Parameter Tuning** | None | β (critical) | k, f | k, f, m | Clustering params |
| **Implementation Complexity** | Low | Low | Medium | Medium | High |
| **Scalability** | High | High | Low | Low | Low |

---

## 7. Suitability for DistribAI

### Use Case Analysis

DistribAI's Byzantine fault tolerance requirements:
- **Scale**: 1000+ nodes (potentially 10,000+)
- **Threat model**: Unknown fraction of malicious nodes, potential collusion
- **Heterogeneity**: Non-IID data, varying hardware
- **Performance**: Must not significantly slow training
- **Adaptability**: Should adapt to changing threat landscape

### Method Suitability

**Coordinate-wise Median:**
- **Pros**: Fast, scalable, simple
- **Cons**: Vulnerable to multi-dimensional attacks
- **Verdict**: Good baseline, but insufficient for sophisticated attacks

**Trimmed Mean:**
- **Pros**: Fast, tunable
- **Cons**: Requires accurate β estimate, sensitive to non-IID data
- **Verdict**: Good if Byzantine fraction is known, but risky for unknown threats

**Krum:**
- **Pros**: Strong robustness, multi-dimensional
- **Cons**: O(n²) complexity, slow convergence (single selection)
- **Verdict**: Too slow for 1000+ nodes, unstable convergence

**Multi-Krum:**
- **Pros**: Strong robustness, stable convergence
- **Cons**: O(n²) complexity, still expensive for large n
- **Verdict**: Best robustness, but scalability concerns

**AUROR:**
- **Pros**: Unsupervised, adaptive
- **Cons**: Assumption-dependent, expensive clustering
- **Verdict**: Good for unknown threats, but scalability concerns

---

## 8. Recommended Strategy

### Primary Recommendation: Hybrid Approach

**Phase 0 (Development - 10 nodes):**
- Use **Coordinate-wise Median** as baseline
- Simple, fast, sufficient for small scale
- Establish baseline performance

**Phase 1 (Alpha - 50 nodes):**
- Implement **Trimmed Mean** with adaptive β
- Start with β = 0.1 (assume 10% Byzantine)
- Dynamically adjust β based on outlier detection
- Monitor convergence and accuracy

**Phase 2 (Beta - 200 nodes):**
- Implement **Multi-Krum** with optimized distance calculations
- Use approximation algorithms (e.g., locality-sensitive hashing) to reduce O(n²) complexity
- Combine with Trimmed Mean for two-stage filtering:
  1. Trimmed Mean removes extreme outliers
  2. Multi-Krum selects central updates
- Ablation study to validate hybrid approach

**Phase 3+ (Production - 1000+ nodes):**
- Implement **Hierarchical Multi-Krum**:
  - Cluster nodes into groups of ~100
  - Run Multi-Krum within each group
  - Aggregate group representatives with another Multi-Krum
  - Reduces complexity from O(n²) to O(n log n)
- Add **AUROR** as fallback for unknown attack patterns
- Implement adaptive strategy selection based on threat level

### Alternative: Tiered Approach

If computational resources are limited:
- **Tier 1 nodes** (trusted, verified): Use simple averaging
- **Tier 2 nodes** (unverified): Use Trimmed Mean
- **Tier 3 nodes** (new/suspicious): Use Multi-Krum
- Reduces computational overhead while maintaining security

---

## 9. Implementation Details

### Trimmed Mean Implementation

```go
func TrimmedMean(updates [][]float64, beta float64) []float64 {
    n := len(updates)
    d := len(updates[0])
    result := make([]float64, d)
    
    trimCount := int(beta * float64(n))
    
    for j := 0; j < d; j++ {
        // Extract j-th dimension from all updates
        values := make([]float64, n)
        for i := 0; i < n; i++ {
            values[i] = updates[i][j]
        }
        
        // Sort values
        sort.Float64s(values)
        
        // Trim smallest and largest
        trimmed := values[trimCount : n-trimCount]
        
        // Compute mean of remaining
        sum := 0.0
        for _, v := range trimmed {
            sum += v
        }
        result[j] = sum / float64(len(trimmed))
    }
    
    return result
}
```

### Multi-Krum Implementation (Optimized)

```go
func MultiKrum(updates [][]float64, f int, m int) []float64 {
    n := len(updates)
    k := n - f - 2
    
    // Compute pairwise distances (optimized with caching)
    scores := computeKrumScores(updates, k)
    
    // Select m updates with lowest scores
    selected := selectLowestScores(updates, scores, m)
    
    // Average selected updates
    return meanUpdates(selected)
}

func computeKrumScores(updates [][]float64, k int) []float64 {
    n := len(updates)
    scores := make([]float64, n)
    
    for i := 0; i < n; i++ {
        // Compute distances to all other updates
        distances := make([]float64, n)
        for j := 0; j < n; j++ {
            if i != j {
                distances[j] = euclideanDistance(updates[i], updates[j])
            }
        }
        
        // Find k nearest neighbors
        sort.Float64s(distances)
        nearestK := distances[1 : k+1] // Skip self (distance = 0)
        
        // Sum squared distances
        sum := 0.0
        for _, d := range nearestK {
            sum += d * d
        }
        scores[i] = sum
    }
    
    return scores
}
```

### Hierarchical Multi-Krum

```go
func HierarchicalMultiKrum(updates [][]float64, f int, m int, groupSize int) []float64 {
    n := len(updates)
    
    // If small enough, use standard Multi-Krum
    if n <= groupSize {
        return MultiKrum(updates, f, m)
    }
    
    // Divide into groups
    numGroups := (n + groupSize - 1) / groupSize
    groupRepresentatives := make([][]float64, numGroups)
    
    for g := 0; g < numGroups; g++ {
        start := g * groupSize
        end := min((g+1)*groupSize, n)
        groupUpdates := updates[start:end]
        
        // Run Multi-Krum within group
        groupRepresentatives[g] = MultiKrum(groupUpdates, f/numGroups, m)
    }
    
    // Aggregate group representatives
    return MultiKrum(groupRepresentatives, numGroups, min(m, numGroups))
}
```

---

## 10. Adaptive β Estimation

For Trimmed Mean, dynamically estimate β based on outlier detection:

```go
func EstimateBeta(updates [][]float64, currentBeta float64) float64 {
    n := len(updates)
    d := len(updates[0])
    
    // Count outliers per dimension using current beta
    outlierCount := 0
    for j := 0; j < d; j++ {
        values := make([]float64, n)
        for i := 0; i < n; i++ {
            values[i] = updates[i][j]
        }
        
        sort.Float64s(values)
        trimCount := int(currentBeta * float64(n))
        trimmed := values[trimCount : n-trimCount]
        
        // Count values outside trimmed range
        for i := 0; i < n; i++ {
            if values[i] < trimmed[0] || values[i] > trimmed[len(trimmed)-1] {
                outlierCount++
            }
        }
    }
    
    // Adjust beta based on outlier ratio
    outlierRatio := float64(outlierCount) / float64(n * d)
    newBeta := min(0.4, max(0.05, outlierRatio * 2)) // Clamp between 5% and 40%
    
    // Smooth adjustment (exponential moving average)
    return 0.9 * currentBeta + 0.1 * newBeta
}
```

---

## 11. Integration with Aggregation Strategy

Byzantine fault tolerance should be integrated with the aggregation strategy from RFC 005:

```
1. Receive gradients from n nodes
2. Apply Byzantine detection:
   - Phase 0-1: Trimmed Mean with adaptive β
   - Phase 2+: Hierarchical Multi-Krum
3. Filter out detected Byzantine updates
4. Apply aggregation strategy:
   - FedAvg (baseline)
   - FedProx (for non-IID data)
   - Scaffold (for variance reduction)
5. Update global model
6. Update node reliability scores
7. Ban nodes with repeated Byzantine behavior
```

### Node Reliability Scoring

Track node reliability over time:

```go
type NodeReliability struct {
    NodeID        string
    SuccessCount  int
    FailureCount  int
    Reliability   float64 // 0.0 to 1.0
    LastUpdate    time.Time
}

func UpdateReliability(node *NodeReliability, isByzantine bool) {
    if isByzantine {
        node.FailureCount++
    } else {
        node.SuccessCount++
    }
    
    // Exponential moving average
    alpha := 0.1
    newReliability := float64(node.SuccessCount) / float64(node.SuccessCount + node.FailureCount)
    node.Reliability = alpha * newReliability + (1 - alpha) * node.Reliability
    node.LastUpdate = time.Now()
}

func ShouldBan(node *NodeReliability) bool {
    return node.Reliability < 0.3 && (node.SuccessCount + node.FailureCount) > 10
}
```

---

## 12. Open Questions

1. **Scalability**: Can hierarchical Multi-Krum scale to 10,000+ nodes with acceptable latency?
2. **Adaptive β**: What is the optimal adaptation rate for β estimation?
3. **Clustering algorithm**: Which clustering algorithm works best for AUROR in practice?
4. **Attack detection**: How to distinguish between Byzantine attacks and legitimate non-IID variations?
5. **Resource constraints**: Can these methods run on resource-constrained orchestrator hardware?
6. **Byzantine fraction**: What is a reasonable assumption for maximum Byzantine fraction in practice?

---

## 13. Conclusion

Byzantine fault tolerance is critical for DistribAI's security and reliability. The recommended approach is:

1. **Phase 0-1**: Trimmed Mean with adaptive β (simple, fast, tunable)
2. **Phase 2+**: Hierarchical Multi-Krum (strong robustness, scalable)
3. **Fallback**: AUROR for unknown attack patterns
4. **Integration**: Combine with aggregation strategy (FedProx/Scaffold)
5. **Monitoring**: Track node reliability and ban repeat offenders

This hybrid approach provides strong robustness against sophisticated attacks while maintaining scalability for 1000+ nodes. The phased implementation allows gradual complexity increase as the system scales.

**Next step**: Proceed to RFC 007 on Hash-Chained Ledger Implementations.
