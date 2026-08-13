//! Katz centrality: influence via damped path counting.
//!
//! # Intuition
//!
//! Katz centrality counts all paths from other nodes, with longer paths
//! weighted less. Unlike eigenvector centrality, even isolated nodes
//! get a non-zero baseline score.
//!
//! Think of it as: "How many ways can information reach this node,
//! accounting for attenuation over distance?"
//!
//! # Definition
//!
//! ```text
//! x_v = α × Σ_{u→v} x_u + β
//! ```
//!
//! In matrix form: x = α A^T x + β 1
//!
//! Where:
//! - α (alpha) = attenuation factor for each hop
//! - β (beta) = baseline centrality for each node
//!
//! # Relationship to Eigenvector Centrality
//!
//! As α → 1/λ_max (largest eigenvalue), Katz → Eigenvector centrality.
//! The β term prevents instability when α is near this limit.
//!
//! # Choosing α
//!
//! Must satisfy α < 1/λ_max for convergence. Common choices:
//! - α = 0.1 (strong damping, local focus)
//! - α = 0.5 (moderate)
//! - α = 0.85 / λ_max (similar to PageRank damping)
//!
//! # Algorithm
//!
//! Power iteration: x^(k+1) = α A^T x^(k) + β 1
//!
//! Converges because α < 1/λ_max ensures the iteration is contractive.
//!
//! # References
//!
//! - Katz (1953). "A new status index derived from sociometric analysis"

use crate::{EntityId, KnowledgeGraph};
use std::collections::HashMap;

/// Configuration for Katz centrality.
#[derive(Debug, Clone, Copy)]
pub struct KatzConfig {
    /// Attenuation factor per hop. Must be < 1/λ_max for convergence.
    /// Default 0.1 is conservative.
    pub alpha: f64,
    /// Baseline centrality for each node.
    pub beta: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Treat graph as undirected.
    pub undirected: bool,
    /// Normalize final scores.
    pub normalized: bool,
}

impl Default for KatzConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            beta: 1.0,
            max_iterations: 100,
            tolerance: 1e-6,
            undirected: false,
            normalized: true,
        }
    }
}

/// Compute Katz centrality via power iteration.
///
/// # Complexity
///
/// - Time: O(E log d_max + E × iterations) with parallel-edge deduplication
/// - Space: O(V + E)
///
/// # Example
///
/// ```
/// use lattix::{KnowledgeGraph, Triple};
/// use lattix::algo::centrality::{katz_centrality, KatzConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// // Chain: A -> B -> C
/// kg.add_triple(Triple::new("A", "rel", "B"));
/// kg.add_triple(Triple::new("B", "rel", "C"));
///
/// let scores = katz_centrality(&kg, KatzConfig::default());
/// // C receives paths from both A and B
/// // B receives paths from A only
/// // A receives no paths
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn katz_centrality(kg: &KnowledgeGraph, config: KatzConfig) -> HashMap<EntityId, f64> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    // α must be in (0, 1) for the power iteration to converge.
    // The true requirement is α < 1/λ_max, but computing λ_max is expensive;
    // α >= 1 is always invalid regardless of graph structure.
    assert!(
        config.alpha > 0.0 && config.alpha < 1.0,
        "Katz alpha must be in (0, 1) for convergence, got {}",
        config.alpha
    );

    // Initialize with beta
    let mut scores = vec![config.beta; n];
    let mut new_scores = vec![0.0; n];
    let predecessors: Vec<Vec<_>> = graph
        .node_indices()
        .map(|idx| {
            let neighbors = if config.undirected {
                crate::algo::unique_neighbors_undirected(graph, idx)
            } else {
                crate::algo::unique_neighbors_directed(graph, idx, petgraph::Direction::Incoming)
            };
            neighbors
                .into_iter()
                .map(|neighbor| neighbor.index())
                .collect()
        })
        .collect();

    for _iter in 0..config.max_iterations {
        // x^(k+1) = α A^T x^(k) + β
        for (idx, neighbors) in predecessors.iter().enumerate() {
            // α × Σ x_pred + β
            let pred_sum: f64 = neighbors.iter().map(|&pred| scores[pred]).sum();
            new_scores[idx] = config.alpha * pred_sum + config.beta;
        }

        // Check convergence
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();

        std::mem::swap(&mut scores, &mut new_scores);

        if diff < config.tolerance {
            break;
        }
    }

    // Normalize if requested
    if config.normalized {
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max > 0.0 {
            for s in &mut scores {
                *s /= max;
            }
        }
    }

    // Map to entity IDs
    graph
        .node_indices()
        .map(|idx| (graph[idx].id.clone(), scores[idx.index()]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_katz_chain() {
        let mut kg = KnowledgeGraph::new();
        // Chain: A -> B -> C
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let config = KatzConfig {
            normalized: false,
            ..Default::default()
        };
        let scores = katz_centrality(&kg, config);

        let a = *scores.get("A").unwrap();
        let b = *scores.get("B").unwrap();
        let c = *scores.get("C").unwrap();

        // C gets paths from A and B, B gets from A, A gets baseline only
        // With α=0.1, β=1:
        // A = 1 (baseline only)
        // B = 0.1 * 1 + 1 = 1.1
        // C = 0.1 * 1.1 + 1 = 1.11
        assert!(c > b, "C={c} should be > B={b}");
        assert!(b > a, "B={b} should be > A={a}");
    }

    #[test]
    fn test_katz_isolated() {
        let mut kg = KnowledgeGraph::new();
        // Disconnected nodes
        kg.add_triple(Triple::new("A", "rel", "B"));
        // C and D are isolated (only entity through other triple)
        kg.add_triple(Triple::new("C", "rel", "D"));

        let config = KatzConfig::default();
        let scores = katz_centrality(&kg, config);

        // All nodes have at least baseline score
        for (name, score) in &scores {
            assert!(*score > 0.0, "{name} should have positive score: {score}");
        }
    }

    #[test]
    fn test_katz_vs_eigenvector() {
        // Katz with β=0 and small α should behave like eigenvector
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));

        let config = KatzConfig {
            alpha: 0.1,
            beta: 0.0,
            normalized: true,
            ..Default::default()
        };
        let scores = katz_centrality(&kg, config);

        // B should be most central (connected to both A and C)
        let b = *scores.get("B").unwrap();
        let a = *scores.get("A").unwrap();
        let c = *scores.get("C").unwrap();

        assert!(b >= a, "B={b} should be >= A={a}");
        assert!(b >= c, "B={b} should be >= C={c}");
    }

    #[test]
    fn test_katz_normalized() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let config = KatzConfig {
            normalized: true,
            ..Default::default()
        };
        let scores = katz_centrality(&kg, config);

        // Max should be 1.0 after normalization
        let max = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 1.0).abs() < 1e-6, "Max should be 1.0: {max}");
    }
}
