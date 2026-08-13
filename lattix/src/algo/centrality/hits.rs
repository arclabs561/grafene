//! HITS algorithm: hubs and authorities.
//!
//! # Intuition
//!
//! HITS (Hyperlink-Induced Topic Search) distinguishes two kinds of importance:
//!
//! - **Authorities**: Nodes that contain valuable content (cited by many)
//! - **Hubs**: Nodes that point to valuable content (cite good sources)
//!
//! Good hubs point to good authorities.
//! Good authorities are pointed to by good hubs.
//!
//! # The Web Search Origin
//!
//! Kleinberg (1999) designed HITS for web search:
//! - A Wikipedia article on "Python" is an **authority** (cited often)
//! - A curated list of Python resources is a **hub** (cites good sources)
//!
//! # Algorithm
//!
//! Iteratively refine hub and authority scores:
//!
//! ```text
//! auth(v) = Σ_{u→v} hub(u)     (authorities = sum of incoming hub scores)
//! hub(v) = Σ_{v→u} auth(u)     (hubs = sum of outgoing authority scores)
//! ```
//!
//! In matrix form:
//! ```text
//! a = A^T × h
//! h = A × a
//! ```
//!
//! Equivalent to finding principal eigenvectors of A^T A (authorities)
//! and A A^T (hubs).
//!
//! # Comparison to PageRank
//!
//! | Aspect | PageRank | HITS |
//! |--------|----------|------|
//! | Scores | Single score | Hub + Authority |
//! | Damping | Yes | No |
//! | Query-specific | No | Originally yes |
//! | Use case | General ranking | Topical search |
//!
//! # References
//!
//! - Kleinberg (1999). "Authoritative sources in a hyperlinked environment"

use crate::{EntityId, KnowledgeGraph};
use std::collections::HashMap;

/// Configuration for HITS algorithm.
#[derive(Debug, Clone, Copy)]
pub struct HitsConfig {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Normalize scores to sum to 1.
    pub normalized: bool,
}

impl Default for HitsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            normalized: true,
        }
    }
}

/// HITS scores for a node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitsScores {
    /// Hub score: how well this node points to authorities.
    pub hub: f64,
    /// Authority score: how well this node is pointed to by hubs.
    pub authority: f64,
}

/// Compute HITS hub and authority scores.
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
/// use lattix::algo::centrality::{hits, HitsConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// // Hub pattern: H1 -> A, H1 -> B, H2 -> A, H2 -> B
/// kg.add_triple(Triple::new("H1", "rel", "A"));
/// kg.add_triple(Triple::new("H1", "rel", "B"));
/// kg.add_triple(Triple::new("H2", "rel", "A"));
/// kg.add_triple(Triple::new("H2", "rel", "B"));
///
/// let scores = hits(&kg, HitsConfig::default());
/// // H1, H2 are hubs (high hub score)
/// // A, B are authorities (high authority score)
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn hits(kg: &KnowledgeGraph, config: HitsConfig) -> HashMap<EntityId, HitsScores> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    // Initialize uniformly
    let init_val = 1.0 / (n as f64).sqrt();
    let mut hub = vec![init_val; n];
    let mut auth = vec![init_val; n];
    let mut new_hub = vec![0.0; n];
    let mut new_auth = vec![0.0; n];
    let incoming: Vec<Vec<_>> = graph
        .node_indices()
        .map(|idx| {
            crate::algo::unique_neighbors_directed(graph, idx, petgraph::Direction::Incoming)
                .into_iter()
                .map(|neighbor| neighbor.index())
                .collect()
        })
        .collect();
    let outgoing: Vec<Vec<_>> = graph
        .node_indices()
        .map(|idx| {
            crate::algo::unique_neighbors_directed(graph, idx, petgraph::Direction::Outgoing)
                .into_iter()
                .map(|neighbor| neighbor.index())
                .collect()
        })
        .collect();

    for _iter in 0..config.max_iterations {
        // Step 1: Update authority scores
        // auth(v) = Σ_{u→v} hub(u)
        new_auth.fill(0.0);
        for (idx, predecessors) in incoming.iter().enumerate() {
            for &pred in predecessors {
                new_auth[idx] += hub[pred];
            }
        }

        // Normalize authority scores
        let auth_norm: f64 = new_auth.iter().map(|x| x * x).sum::<f64>().sqrt();
        if auth_norm > 0.0 {
            for a in &mut new_auth {
                *a /= auth_norm;
            }
        }

        // Step 2: Update hub scores
        // hub(v) = Σ_{v→u} auth(u)
        new_hub.fill(0.0);
        for (idx, successors) in outgoing.iter().enumerate() {
            for &succ in successors {
                new_hub[idx] += new_auth[succ];
            }
        }

        // Normalize hub scores
        let hub_norm: f64 = new_hub.iter().map(|x| x * x).sum::<f64>().sqrt();
        if hub_norm > 0.0 {
            for h in &mut new_hub {
                *h /= hub_norm;
            }
        }

        // Check convergence
        let hub_diff: f64 = hub
            .iter()
            .zip(new_hub.iter())
            .map(|(old, new)| (old - new).powi(2))
            .sum::<f64>()
            .sqrt();

        let auth_diff: f64 = auth
            .iter()
            .zip(new_auth.iter())
            .map(|(old, new)| (old - new).powi(2))
            .sum::<f64>()
            .sqrt();

        std::mem::swap(&mut hub, &mut new_hub);
        std::mem::swap(&mut auth, &mut new_auth);

        if hub_diff < config.tolerance && auth_diff < config.tolerance {
            break;
        }
    }

    // Optional: normalize to sum to 1
    if config.normalized {
        let hub_sum: f64 = hub.iter().sum();
        let auth_sum: f64 = auth.iter().sum();
        if hub_sum > 0.0 {
            for h in &mut hub {
                *h /= hub_sum;
            }
        }
        if auth_sum > 0.0 {
            for a in &mut auth {
                *a /= auth_sum;
            }
        }
    }

    // Map to entity IDs
    graph
        .node_indices()
        .map(|idx| {
            let i = idx.index();
            (
                graph[idx].id.clone(),
                HitsScores {
                    hub: hub[i],
                    authority: auth[i],
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_hits_hub_authority_pattern() {
        let mut kg = KnowledgeGraph::new();
        // Clear hub/authority pattern
        // Hubs: H1, H2 (point to authorities)
        // Authorities: A1, A2 (pointed to by hubs)
        kg.add_triple(Triple::new("H1", "rel", "A1"));
        kg.add_triple(Triple::new("H1", "rel", "A2"));
        kg.add_triple(Triple::new("H2", "rel", "A1"));
        kg.add_triple(Triple::new("H2", "rel", "A2"));

        let scores = hits(&kg, HitsConfig::default());

        let h1 = scores.get("H1").unwrap();
        let a1 = scores.get("A1").unwrap();

        // Hubs should have high hub scores, low authority
        // Authorities should have high authority scores, low hub
        assert!(
            h1.hub > h1.authority,
            "H1 hub={} should be > authority={}",
            h1.hub,
            h1.authority
        );
        assert!(
            a1.authority > a1.hub,
            "A1 authority={} should be > hub={}",
            a1.authority,
            a1.hub
        );
    }

    #[test]
    fn test_hits_symmetric() {
        let mut kg = KnowledgeGraph::new();
        // Symmetric: everyone points to everyone
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let config = HitsConfig {
            normalized: false,
            ..Default::default()
        };
        let scores = hits(&kg, config);

        let a = scores.get("A").unwrap();
        let b = scores.get("B").unwrap();

        // In symmetric graph, all should be roughly equal
        assert!(
            (a.hub - b.hub).abs() < 0.1,
            "Hub scores should be similar: A={}, B={}",
            a.hub,
            b.hub
        );
        assert!(
            (a.authority - b.authority).abs() < 0.1,
            "Auth scores should be similar: A={}, B={}",
            a.authority,
            b.authority
        );
    }

    #[test]
    fn test_hits_chain() {
        let mut kg = KnowledgeGraph::new();
        // Chain: A -> B -> C
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let scores = hits(&kg, HitsConfig::default());

        let a = scores.get("A").unwrap();
        let b = scores.get("B").unwrap();
        let c = scores.get("C").unwrap();

        // A is a hub (points to B)
        // C is an authority (pointed to by B)
        // B is both
        assert!(a.hub > 0.0, "A should have hub score: {}", a.hub);
        assert!(
            c.authority > 0.0,
            "C should have authority score: {}",
            c.authority
        );
        assert!(
            b.hub > 0.0 && b.authority > 0.0,
            "B should have both: hub={}, auth={}",
            b.hub,
            b.authority
        );
    }

    #[test]
    fn test_hits_normalized_sums() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let scores = hits(
            &kg,
            HitsConfig {
                normalized: true,
                ..Default::default()
            },
        );

        let hub_sum: f64 = scores.values().map(|s| s.hub).sum();
        let auth_sum: f64 = scores.values().map(|s| s.authority).sum();

        assert!(
            (hub_sum - 1.0).abs() < 1e-6,
            "Hub scores should sum to 1: {hub_sum}"
        );
        assert!(
            (auth_sum - 1.0).abs() < 1e-6,
            "Auth scores should sum to 1: {auth_sum}"
        );
    }
}
