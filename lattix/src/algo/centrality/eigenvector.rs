//! Eigenvector centrality: importance via important neighbors.
//!
//! # Intuition
//!
//! A node is important if it's connected to other important nodes.
//! This creates a recursive definition resolved by finding the
//! dominant eigenvector of the adjacency matrix.
//!
//! In a social network: popular people connected to popular people.
//! In a citation network: papers cited by influential papers.
//!
//! # Definition
//!
//! ```text
//! x_v = (1/λ) × Σ_{u→v} x_u
//! ```
//!
//! Equivalently: Ax = λx, where A is the adjacency matrix and λ is
//! the largest eigenvalue (spectral radius).
//!
//! # Algorithm: Power Iteration
//!
//! 1. Initialize x uniformly
//! 2. Repeat: x' = A × x, then normalize x' = x' / ||x'||
//! 3. Stop when ||x' - x|| < tolerance
//!
//! Converges to the dominant eigenvector by the Perron-Frobenius theorem
//! (assuming the graph is strongly connected and aperiodic).
//!
//! # Comparison to PageRank
//!
//! | Aspect | Eigenvector | PageRank |
//! |--------|-------------|----------|
//! | Damping | None | 0.85 (teleportation) |
//! | Dangling nodes | Can cause non-convergence | Redistributed |
//! | Isolated components | Undefined behavior | Handle gracefully |
//!
//! For general graphs, PageRank is more robust. Eigenvector centrality
//! is best for strongly connected graphs.
//!
//! # References
//!
//! - Bonacich (1972). "Factoring and weighting approaches to status scores"
//! - Bonacich (1987). "Power and centrality: A family of measures"

use crate::{EntityId, KnowledgeGraph};
use std::collections::HashMap;

/// Configuration for eigenvector centrality.
#[derive(Debug, Clone, Copy)]
pub struct EigenvectorConfig {
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Convergence tolerance (L2 norm of change).
    pub tolerance: f64,
    /// Treat graph as undirected.
    pub undirected: bool,
}

impl Default for EigenvectorConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            undirected: false,
        }
    }
}

/// Compute eigenvector centrality via power iteration.
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
/// use lattix::algo::centrality::{eigenvector_centrality, EigenvectorConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// // Mutual pointing: A <-> B <-> C <-> A
/// kg.add_triple(Triple::new("A", "rel", "B"));
/// kg.add_triple(Triple::new("B", "rel", "A"));
/// kg.add_triple(Triple::new("B", "rel", "C"));
/// kg.add_triple(Triple::new("C", "rel", "B"));
/// kg.add_triple(Triple::new("C", "rel", "A"));
/// kg.add_triple(Triple::new("A", "rel", "C"));
///
/// let scores = eigenvector_centrality(&kg, EigenvectorConfig::default());
/// // Symmetric graph: all nodes should have equal centrality
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn eigenvector_centrality(
    kg: &KnowledgeGraph,
    config: EigenvectorConfig,
) -> HashMap<EntityId, f64> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    // Initialize uniformly
    let init_val = 1.0 / (n as f64).sqrt();
    let mut scores = vec![init_val; n];
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
        // Compute A × x (or A^T × x for in-edge centrality)
        new_scores.fill(0.0);

        for (idx, neighbors) in predecessors.iter().enumerate() {
            for &pred in neighbors {
                new_scores[idx] += scores[pred];
            }
        }

        // Normalize (L2 norm)
        let norm: f64 = new_scores.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for s in &mut new_scores {
                *s /= norm;
            }
        } else {
            // All zeros - graph has no edges or is problematic
            // Fall back to uniform
            let uniform = 1.0 / (n as f64).sqrt();
            new_scores.fill(uniform);
        }

        // Check convergence
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).powi(2))
            .sum::<f64>()
            .sqrt();

        std::mem::swap(&mut scores, &mut new_scores);

        if diff < config.tolerance {
            break;
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
    fn test_eigenvector_symmetric() {
        let mut kg = KnowledgeGraph::new();
        // Symmetric triangle
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));
        kg.add_triple(Triple::new("C", "rel", "A"));
        kg.add_triple(Triple::new("A", "rel", "C"));

        let scores = eigenvector_centrality(&kg, EigenvectorConfig::default());

        let a = *scores.get("A").unwrap();
        let b = *scores.get("B").unwrap();
        let c = *scores.get("C").unwrap();

        // All should be approximately equal
        assert!((a - b).abs() < 0.01, "A={a}, B={b} should be equal");
        assert!((b - c).abs() < 0.01, "B={b}, C={c} should be equal");
    }

    #[test]
    fn test_eigenvector_star() {
        let mut kg = KnowledgeGraph::new();
        // Star with bidirectional edges
        kg.add_triple(Triple::new("Hub", "rel", "A"));
        kg.add_triple(Triple::new("A", "rel", "Hub"));
        kg.add_triple(Triple::new("Hub", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "Hub"));
        kg.add_triple(Triple::new("Hub", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "Hub"));

        let scores = eigenvector_centrality(&kg, EigenvectorConfig::default());

        let hub = *scores.get("Hub").unwrap();
        let a = *scores.get("A").unwrap();

        // Hub should have higher centrality (connected to all)
        assert!(hub > a, "Hub={hub} should be more central than A={a}");
    }

    #[test]
    fn test_eigenvector_normalized() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));

        let scores = eigenvector_centrality(&kg, EigenvectorConfig::default());

        // L2 norm should be 1
        let norm: f64 = scores.values().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Scores should be L2 normalized: {norm}"
        );
    }
}
