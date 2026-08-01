# RFC 007: Hash-Chained Ledger Implementations

**Status:** ✅ Complete  
**Date:** 2026-04-21  
**Research Task:** Hash-Chained Ledger Implementations  
**Output:** docs/rfcs/007-hash-chained-ledger.md

---

## Executive Summary

This RFC analyzes hash-chained ledger implementations for creating tamper-evident, immutable records in distributed systems. Key technologies include **hash chains**, **Merkle trees**, and **Merkle proofs**. The research indicates that **Merkle tree-based append-only logs** (similar to Google's Trillian) provide the best balance of efficiency, verifiability, and scalability for DistribAI's needs for immutable credit tracking and audit trails.

---

## 1. Problem Statement

### Need for Immutable Records

DistribAI requires immutable, tamper-evident records for:
- **Credit tracking**: Ensure contributor credits cannot be manipulated
- **Audit trails**: Log all training jobs, gradient updates, and system events
- **Model provenance**: Track model checkpoints and their lineage
- **Dispute resolution**: Provide cryptographic proof of system state at any time
- **Regulatory compliance**: Enable auditors to verify system integrity

### Requirements
- **Append-only**: Records can only be added, never modified or deleted
- **Tamper-evident**: Any modification is detectable
- **Efficient verification**: Verify inclusion without downloading entire log
- **Scalable**: Support millions of records
- **Cryptographic**: Use standard cryptographic primitives (SHA-256, etc.)

---

## 2. Hash Chains

### Overview
A hash chain is a sequence of records where each record contains the hash of the previous record, creating a cryptographically linked chain.

### Structure
```
Record 0: {data: "initial", prev_hash: null, hash: H0}
Record 1: {data: "tx1", prev_hash: H0, hash: H1}
Record 2: {data: "tx2", prev_hash: H1, hash: H2}
Record 3: {data: "tx3", prev_hash: H2, hash: H3}
...
```

Where `H_i = hash(data_i || prev_hash_i)`

### Properties
- **Append-only**: Cannot insert or reorder records without breaking chain
- **Tamper-evident**: Any modification changes all subsequent hashes
- **Simple**: Easy to implement
- **Linear verification**: Must verify entire chain to confirm integrity

### Advantages
- **Simple implementation**: Straightforward data structure
- **Strong integrity**: Any tampering is detectable
- **Minimal overhead**: Only stores hash of previous record

### Disadvantages
- **Linear verification time**: O(n) to verify entire chain
- **No efficient inclusion proofs**: Cannot prove specific record without full chain
- **Poor scalability**: Verification time grows with log size
- **Sequential access**: Cannot skip to arbitrary records

### Suitability for DistribAI
- **Low**: Too slow for large-scale verification
- **Best for**: Small logs (<10,000 records) with simple requirements
- **Worst for**: Large-scale systems requiring efficient verification

---

## 3. Merkle Trees

### Overview
A Merkle tree (hash tree) is a hierarchical data structure where leaf nodes are hashes of data blocks, and non-leaf nodes are hashes of their children. The root hash serves as a fingerprint for the entire dataset.

### Structure
```
                    Root Hash (H_ABCD)
                     /              \
                H_AB                  H_CD
               /    \               /    \
            H_A      H_B         H_C      H_D
            /        \           /        \
          Data A    Data B    Data C    Data D
```

Where:
- `H_A = hash(Data A)`
- `H_B = hash(Data B)`
- `H_AB = hash(H_A || H_B)`
- `Root = hash(H_AB || H_CD)`

### Properties
- **Append-only**: Adding new leaves changes root hash
- **Tamper-evident**: Any change to data propagates to root
- **Efficient verification**: O(log n) inclusion proofs
- **Scalable**: Verification time grows logarithmically

### Advantages
- **Efficient inclusion proofs**: Verify specific record without full dataset
- **Logarithmic verification**: O(log n) vs O(n) for hash chains
- **Parallelizable**: Can verify multiple branches independently
- **Standardized**: Well-understood, widely used (blockchain, Git)
- **Space-efficient**: Only need root hash for integrity verification

### Disadvantages
- **Complexity**: More complex than simple hash chains
- **Tree construction overhead**: Must build tree for each batch
- **Memory overhead**: Must store tree structure (though minimal)
- **Rebuild on append**: Adding to tree requires recomputing path to root

### Suitability for DistribAI
- **High**: Excellent balance of efficiency and verifiability
- **Best for**: Large-scale systems requiring efficient verification
- **Worst for**: Simple use cases where hash chain suffices

### Merkle Proofs

A Merkle proof verifies that a specific data block is part of the dataset without requiring the entire dataset.

**Example:**
To verify that Data A is in the tree:
1. User knows: Data A, Root Hash (H_ABCD)
2. Full node provides: H_B, H_CD
3. User computes:
   - `H_A = hash(Data A)`
   - `H_AB = hash(H_A || H_B)`
   - `H_ABCD = hash(H_AB || H_CD)`
4. Compare computed H_ABCD with known root hash
5. If match: Data A is in the tree

**Efficiency:**
- For n leaves: O(log n) hashes in proof
- For 1,024 leaves: ~10 hashes
- For 1,000,000 leaves: ~20 hashes

---

## 4. Append-Only Logs (Trillian-style)

### Overview
Trillian is Google's open-source implementation of a tamper-evident log using Merkle trees. It provides an append-only log with cryptographic verification of:
- Record inclusion (record is in the log)
- Log integrity (log hasn't been tampered with)
- Consistency (log is consistent with previous state)

### Architecture
```
Log Entry 1 → Merkle Tree → Root Hash 1 → Signed Head
Log Entry 2 → Merkle Tree → Root Hash 2 → Signed Head
Log Entry 3 → Merkle Tree → Root Hash 3 → Signed Head
...
```

### Key Features
- **Append-only**: Records can only be added, never modified
- **Merkle tree**: Efficient O(log n) inclusion proofs
- **Signed heads**: Each root hash is signed by trusted authority
- **Consistency proofs**: Verify log hasn't been tampered with between two points in time
- **Batch operations**: Can add multiple records in single batch

### Advantages
- **Production-proven**: Powers Certificate Transparency (CT) ecosystem
- **Efficient**: O(log n) verification
- **Flexible**: Supports multiple verifiable data structures
- **Open source**: Apache 2.0 license
- **Scalable**: Designed for large-scale deployments

### Disadvantages
- **Complexity**: More complex than simple hash chain
- **Dependency**: Requires external library (Trillian)
- **Infrastructure**: Requires dedicated log server
- **Learning curve**: Team must understand Merkle trees and proofs

### Suitability for DistribAI
- **Very High**: Best-in-class solution for tamper-evident logging
- **Best for**: Production systems requiring strong guarantees
- **Worst for**: Simple prototypes or small-scale deployments

---

## 5. Comparative Analysis

| Aspect | Hash Chain | Merkle Tree | Trillian-style Log |
|--------|------------|-------------|-------------------|
| **Verification Time** | O(n) | O(log n) | O(log n) |
| **Inclusion Proof** | No | Yes | Yes |
| **Consistency Proof** | No | Limited | Yes |
| **Implementation Complexity** | Low | Medium | High |
| **Scalability** | Poor | Good | Excellent |
| **Production Readiness** | Low | Medium | High |
| **Memory Overhead** | Minimal | Low (tree) | Medium (tree + metadata) |
| **Append Overhead** | O(1) | O(log n) | O(log n) |
| **Tamper Evidence** | Strong | Strong | Strong |

---

## 6. Suitability for DistribAI

### Use Case Analysis

DistribAI's immutable ledger requirements:
- **Scale**: Millions of records (credits, jobs, checkpoints)
- **Verification**: Contributors must verify credits without downloading full log
- **Auditability**: Regulators must verify system integrity
- **Performance**: Verification must be fast (<100ms)
- **Trust**: Minimal trust assumptions (decentralized verification)

### Method Suitability

**Hash Chain:**
- **Pros**: Simple, minimal overhead
- **Cons**: O(n) verification, no inclusion proofs
- **Verdict**: Insufficient for large-scale DistribAI

**Merkle Tree:**
- **Pros**: O(log n) verification, inclusion proofs
- **Cons**: Must implement from scratch, no consistency proofs
- **Verdict**: Good, but requires custom implementation

**Trillian-style Log:**
- **Pros**: Production-proven, full feature set, consistency proofs
- **Cons**: External dependency, infrastructure complexity
- **Verdict**: Best choice for production DistribAI

---

## 7. Recommended Architecture

### Primary Recommendation: Merkle Tree-based Append-Only Log

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    DistribAI Orchestrator                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Credit Ledger│    │  Job Ledger  │    │ Audit Ledger │ │
│  │  (Merkle)    │    │  (Merkle)    │    │  (Merkle)    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             │                              │
│                    ┌────────▼────────┐                     │
│                    │ Merkle Root Store│                     │
│                    │ (signed heads)   │                     │
│                    └─────────────────┘                     │
│                             │                              │
│                    ┌────────▼────────┐                     │
│                    │  Proof Service  │                     │
│                    │ (inclusion/consistency)               │
│                    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Plan

**Phase 0 (Development - 10 nodes):**
- Implement simple **hash chain** for credit tracking
- Store in database with `prev_hash` column
- Verify integrity on startup
- Establish baseline performance

**Phase 1 (Alpha - 50 nodes):**
- Implement **Merkle tree** for credit ledger
- Add inclusion proof API
- Implement root hash signing
- Separate ledgers: credits, jobs, audits

**Phase 2 (Beta - 200 nodes):**
- Implement **consistency proofs** (verify log hasn't been tampered)
- Add **batch operations** (append multiple records)
- Implement **proof service** for contributor verification
- Add **periodic root hash anchoring** (to public blockchain or trusted timestamp)

**Phase 3+ (Production - 1000+ nodes):**
- Evaluate **Trillian integration** for production use
- Implement **sharded ledgers** (by time period or contributor)
- Add **Merkle proof caching** for performance
- Implement **audit API** for regulators

---

## 8. Implementation Details

### Merkle Tree Implementation (Go)

```go
package ledger

import (
    "crypto/sha256"
    "encoding/hex"
    "errors"
)

type MerkleTree struct {
    Root       *Node
    Leaves     []*Node
    LeafHashes [][]byte
}

type Node struct {
    Left  *Node
    Right *Node
    Hash  []byte
}

func NewMerkleTree(data [][]byte) (*MerkleTree, error) {
    if len(data) == 0 {
        return nil, errors.New("empty data")
    }
    
    // Create leaf nodes
    leaves := make([]*Node, len(data))
    leafHashes := make([][]byte, len(data))
    
    for i, d := range data {
        hash := sha256.Sum256(d)
        leafHashes[i] = hash[:]
        leaves[i] = &Node{Hash: hash[:]}
    }
    
    // Build tree
    root := buildTree(leaves)
    
    return &MerkleTree{
        Root:       root,
        Leaves:     leaves,
        LeafHashes: leafHashes,
    }, nil
}

func buildTree(nodes []*Node) *Node {
    if len(nodes) == 1 {
        return nodes[0]
    }
    
    var nextLevel []*Node
    
    for i := 0; i < len(nodes); i += 2 {
        left := nodes[i]
        var right *Node
        
        if i+1 < len(nodes) {
            right = nodes[i+1]
        } else {
            // Odd number of nodes, duplicate last
            right = left
        }
        
        combined := append(left.Hash, right.Hash...)
        hash := sha256.Sum256(combined)
        
        nextLevel = append(nextLevel, &Node{
            Left:  left,
            Right: right,
            Hash:  hash[:],
        })
    }
    
    return buildTree(nextLevel)
}

func (mt *MerkleTree) RootHash() []byte {
    return mt.Root.Hash
}

func (mt *MerkleTree) GetProof(index int) ([][]byte, error) {
    if index < 0 || index >= len(mt.Leaves) {
        return nil, errors.New("index out of bounds")
    }
    
    var proof [][]byte
    node := mt.Leaves[index]
    
    // Traverse up the tree, collecting sibling hashes
    for node != mt.Root {
        parent := findParent(mt.Root, node)
        if parent == nil {
            break
        }
        
        if parent.Left == node {
            proof = append(proof, parent.Right.Hash)
        } else {
            proof = append(proof, parent.Left.Hash)
        }
        
        node = parent
    }
    
    return proof, nil
}

func findParent(root, node *Node) *Node {
    if root == nil || root == node {
        return nil
    }
    
    if root.Left == node || root.Right == node {
        return root
    }
    
    if parent := findParent(root.Left, node); parent != nil {
        return parent
    }
    
    return findParent(root.Right, node)
}

func VerifyProof(rootHash []byte, data []byte, proof [][]byte, index int) bool {
    computedHash := sha256.Sum256(data)
    
    for i, siblingHash := range proof {
        // Determine order based on index
        if (index>>uint(i))&1 == 0 {
            // Node is on left, sibling on right
            combined := append(computedHash[:], siblingHash...)
            computedHash = sha256.Sum256(combined)
        } else {
            // Node is on right, sibling on left
            combined := append(siblingHash, computedHash[:]...)
            computedHash = sha256.Sum256(combined)
        }
    }
    
    return hex.EncodeToString(computedHash[:]) == hex.EncodeToString(rootHash)
}
```

### Append-Only Ledger with Merkle Tree

```go
package ledger

import (
    "crypto/sha256"
    "encoding/json"
    "time"
)

type Record struct {
    Index     uint64
    Timestamp time.Time
    Data      []byte
    PrevHash  []byte
    Hash      []byte
}

type Ledger struct {
    Records   []Record
    RootHash  []byte
    Signature []byte // Signed by orchestrator
}

func NewLedger() *Ledger {
    return &Ledger{
        Records: make([]Record, 0),
    }
}

func (l *Ledger) Append(data []byte) error {
    var prevHash []byte
    if len(l.Records) > 0 {
        prevHash = l.Records[len(l.Records)-1].Hash
    }
    
    index := uint64(len(l.Records))
    timestamp := time.Now()
    
    // Compute record hash
    recordHash := computeRecordHash(index, timestamp, data, prevHash)
    
    record := Record{
        Index:     index,
        Timestamp: timestamp,
        Data:      data,
        PrevHash:  prevHash,
        Hash:      recordHash,
    }
    
    l.Records = append(l.Records, record)
    
    // Rebuild Merkle tree with all records
    dataHashes := make([][]byte, len(l.Records))
    for i, r := range l.Records {
        dataHashes[i] = r.Hash
    }
    
    tree, err := NewMerkleTree(dataHashes)
    if err != nil {
        return err
    }
    
    l.RootHash = tree.RootHash
    
    // Sign root hash (placeholder)
    l.Signature = signRootHash(l.RootHash)
    
    return nil
}

func computeRecordHash(index uint64, timestamp time.Time, data, prevHash []byte) []byte {
    h := sha256.New()
    
    // Serialize record components
    indexBytes := make([]byte, 8)
    for i := 0; i < 8; i++ {
        indexBytes[i] = byte(index >> (i * 8))
    }
    
    h.Write(indexBytes)
    h.Write([]byte(timestamp.Format(time.RFC3339Nano)))
    h.Write(data)
    h.Write(prevHash)
    
    return h.Sum(nil)
}

func (l *Ledger) GetProof(index uint64) ([][]byte, error) {
    dataHashes := make([][]byte, len(l.Records))
    for i, r := range l.Records {
        dataHashes[i] = r.Hash
    }
    
    tree, err := NewMerkleTree(dataHashes)
    if err != nil {
        return nil, err
    }
    
    return tree.GetProof(int(index))
}

func (l *Ledger) VerifyRecord(index uint64, data []byte) (bool, error) {
    if int(index) >= len(l.Records) {
        return false, errors.New("index out of bounds")
    }
    
    proof, err := l.GetProof(index)
    if err != nil {
        return false, err
    }
    
    return VerifyProof(l.RootHash, data, proof, int(index)), nil
}

func signRootHash(hash []byte) []byte {
    // Placeholder: actual implementation would use orchestrator's private key
    return hash // For now, just return the hash
}
```

### Credit Ledger Specific Implementation

```go
package ledger

type CreditEntry struct {
    ContributorID string
    JobID         string
    Amount        float64
    Timestamp     time.Time
}

type CreditLedger struct {
    *Ledger
}

func NewCreditLedger() *CreditLedger {
    return &CreditLedger{
        Ledger: NewLedger(),
    }
}

func (cl *CreditLedger) AddCredit(contributorID, jobID string, amount float64) error {
    entry := CreditEntry{
        ContributorID: contributorID,
        JobID:         jobID,
        Amount:        amount,
        Timestamp:     time.Now(),
    }
    
    data, err := json.Marshal(entry)
    if err != nil {
        return err
    }
    
    return cl.Append(data)
}

func (cl *CreditLedger) GetCreditProof(index uint64) ([][]byte, error) {
    return cl.GetProof(index)
}

func (cl *CreditLedger) VerifyCredit(index uint64, entry CreditEntry) (bool, error) {
    data, err := json.Marshal(entry)
    if err != nil {
        return false, err
    }
    
    return cl.VerifyRecord(index, data)
}

func (cl *CreditLedger) GetTotalCredits(contributorID string) (float64, error) {
    total := 0.0
    
    for _, record := range cl.Records {
        var entry CreditEntry
        if err := json.Unmarshal(record.Data, &entry); err != nil {
            continue
        }
        
        if entry.ContributorID == contributorID {
            total += entry.Amount
        }
    }
    
    return total, nil
}
```

---

## 9. Integration with DistribAI

### Credit Tracking Workflow

```
1. Contributor completes training job
2. Orchestrator validates job completion
3. Orchestrator adds credit entry to Credit Ledger
4. Ledger computes new Merkle root
5. Orchestrator signs new root hash
6. Contributor receives:
   - Credit amount
   - Merkle proof of credit entry
   - Signed root hash
7. Contributor verifies:
   - Merkle proof against root hash
   - Root hash signature
8. Contributor stores proof locally for future verification
```

### Audit Workflow

```
1. Auditor requests credit ledger state at time T
2. Orchestrator provides:
   - Root hash at time T
   - Signature of root hash
   - Consistency proof from time T-1 to T
3. Auditor verifies:
   - Signature of root hash
   - Consistency proof
   - Selected credit entries with inclusion proofs
```

### Public Anchoring (Optional)

For additional trust, anchor root hashes to public blockchain:

```go
func (cl *CreditLedger) AnchorToBlockchain() (string, error) {
    // Get current root hash
    rootHash := cl.RootHash
    
    // Submit to blockchain (e.g., Ethereum, Bitcoin)
    txHash, err := submitToBlockchain(rootHash)
    if err != nil {
        return "", err
    }
    
    return txHash, nil
}
```

---

## 10. Performance Considerations

### Merkle Tree Size

For n records:
- **Tree nodes**: 2n - 1
- **Memory overhead**: ~64 bytes per node (SHA-256 hash)
- **Total memory**: ~128n bytes

For 1,000,000 records: ~128 MB (acceptable)

### Verification Time

- **Inclusion proof**: O(log n) hash operations
- **For 1,000,000 records**: ~20 hash operations (~1ms)
- **For 10,000,000 records**: ~24 hash operations (~1.2ms)

### Append Overhead

- **Single append**: O(log n) to rebuild path to root
- **Batch append**: O(k log n) for k records
- **Optimization**: Batch appends every 100-1000 records

---

## 11. Security Considerations

### Threats

1. **Rogue orchestrator**: Could tamper with ledger
   - **Mitigation**: Sign root hashes with known public key
   - **Mitigation**: Public anchoring to blockchain

2. **Collusion**: Multiple nodes collude to fake credits
   - **Mitigation**: Require quorum for credit awarding
   - **Mitigation**: Byzantine fault tolerance (RFC 006)

3. **Hash collisions**: Attacker finds collision to fake record
   - **Mitigation**: Use SHA-256 (collision-resistant)
   - **Mitigation**: Use SHA-3 for future-proofing

4. **Replay attacks**: Attacker replays old credits
   - **Mitigation**: Include timestamp and job ID in record
   - **Mitigation**: Check for duplicate job IDs

### Best Practices

- **Use SHA-256**: Industry-standard, collision-resistant
- **Sign root hashes**: Prevent rogue orchestrator tampering
- **Public anchoring**: Optional but provides additional trust
- **Regular audits**: Verify ledger integrity periodically
- **Backup root hashes**: Store in multiple locations

---

## 12. Open Questions

1. **Public anchoring**: Should DistribAI anchor to public blockchain? Which one?
2. **Ledger sharding**: How to shard ledgers for scalability?
3. **Proof caching**: How to cache proofs for performance?
4. **Pruning**: Can old records be pruned while preserving proofs?
5. **Multi-signature**: Should root hashes require multi-signature?

---

## 13. Conclusion

Hash-chained ledgers using Merkle trees provide the best balance of efficiency, verifiability, and scalability for DistribAI's immutable record needs. The recommended approach is:

1. **Phase 0**: Simple hash chain for prototyping
2. **Phase 1**: Merkle tree with inclusion proofs
3. **Phase 2**: Consistency proofs and batch operations
4. **Phase 3+**: Evaluate Trillian integration for production

This approach ensures that credits, job records, and audit trails are tamper-evident and verifiable, providing trust in the DistribAI system without requiring centralized trust.

**Next step**: Complete remaining research tasks and proceed to remove bloat files and push to GitHub fork.
