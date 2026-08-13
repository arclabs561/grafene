//! Closeness centrality: measuring proximity to all other nodes.
//!
//! # Intuition
//!
//! Closeness measures how quickly information can spread from a node.
//! High closeness = short average distance to all others.
//!
//! In a social network: someone who can reach anyone in few hops.
//! In a transit network: a well-connected hub station.
//!
//! # Definition
//!
//! Classic closeness (Bavelas 1950):
//!
//! ```text
//! C_C(v) = (n - 1) / Σ_{u≠v} d(v, u)
//! ```
//!
//! Where d(v, u) is the shortest path distance from v to u.
//!
//! # Handling Disconnected Graphs
//!
//! If some nodes are unreachable, classic closeness breaks (infinite distance).
//! Two common fixes:
//!
//! | Variant | Formula | Behavior |
//! |---------|---------|----------|
//! | **Harmonic** | Σ_{u≠v} 1/d(v,u) | Ignore unreachable (d=∞ → 0) |
//! | **Classic** | r / Σ d(v,u) | Reciprocal mean over `r` reachable peers |
//! | **Wasserman-Faust** | [r/(n-1)] × [r/Σ d(v,u)] | Downweight small components |
//!
//! This implementation uses **harmonic centrality** by default. Classic mode
//! uses the Wasserman-Faust factor when normalization is enabled.
//!
//! # Normalization
//!
//! Harmonic centrality is normalized by dividing by (n-1). For classic
//! closeness, normalization applies the reachable fraction `r/(n-1)`:
//!
//! ```text
//! C_H_norm(v) = C_H(v) / (n - 1)
//! ```
//!
//! # References
//!
//! - Bavelas (1950). "Communication patterns in task-oriented groups"
//! - Rochat (2009). "Closeness centrality extended to unconnected graphs"

use crate::{EntityId, KnowledgeGraph};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, VecDeque};

/// Configuration for closeness centrality.
#[derive(Debug, Clone, Copy)]
pub struct ClosenessConfig {
    /// Normalize harmonic scores by `n - 1`, or apply the
    /// Wasserman-Faust reachable-fraction correction in classic mode.
    pub normalized: bool,
    /// Treat graph as undirected. Directed mode follows outgoing edges.
    pub undirected: bool,
    /// Use harmonic mean (recommended for disconnected graphs).
    pub harmonic: bool,
}

impl Default for ClosenessConfig {
    fn default() -> Self {
        Self {
            normalized: true,
            undirected: false,
            harmonic: true, // robust to disconnected components
        }
    }
}

/// Compute closeness centrality for all nodes.
///
/// Uses harmonic centrality by default, which handles disconnected graphs.
/// Directed scores use outward distances from each node.
///
/// # Complexity
///
/// - Time: O(VE log d_max) (BFS from each node with parallel-edge deduplication)
/// - Space: O(V)
///
/// # Example
///
/// ```
/// use lattix::{KnowledgeGraph, Triple};
/// use lattix::algo::centrality::{closeness_centrality, ClosenessConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// // Star: Hub -> A, Hub -> B, Hub -> C
/// kg.add_triple(Triple::new("Hub", "rel", "A"));
/// kg.add_triple(Triple::new("Hub", "rel", "B"));
/// kg.add_triple(Triple::new("Hub", "rel", "C"));
///
/// let scores = closeness_centrality(&kg, ClosenessConfig::default());
/// // Hub reaches everyone in 1 hop
/// // A, B, C can't reach anyone (directed graph)
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn closeness_centrality(
    kg: &KnowledgeGraph,
    config: ClosenessConfig,
) -> HashMap<EntityId, f64> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n < 2 {
        return graph
            .node_indices()
            .map(|idx| (graph[idx].id.clone(), 0.0))
            .collect();
    }

    let mut result = HashMap::with_capacity(n);

    for source in graph.node_indices() {
        let distances = bfs_distances(graph, source, config.undirected);

        let normalized_closeness = if config.harmonic {
            // Harmonic: Σ 1/d(v,u) for all reachable u
            let sum: f64 = distances
                .iter()
                .enumerate()
                .filter(|(i, &d)| *i != source.index() && d > 0)
                .map(|(_, &d)| 1.0 / d as f64)
                .sum();
            if config.normalized {
                sum / (n - 1) as f64
            } else {
                sum
            }
        } else {
            // Classic: reachable / Σ d(v,u)
            let reachable: Vec<_> = distances
                .iter()
                .enumerate()
                .filter(|(i, &d)| *i != source.index() && d > 0)
                .collect();

            if reachable.is_empty() {
                0.0
            } else {
                let total_dist: i32 = reachable.iter().map(|(_, &d)| d).sum();
                let reachable_count = reachable.len() as f64;
                let classic = reachable_count / f64::from(total_dist);
                if config.normalized {
                    classic * reachable_count / (n - 1) as f64
                } else {
                    classic
                }
            }
        };

        let entity = &graph[source];
        result.insert(entity.id.clone(), normalized_closeness);
    }

    result
}

/// BFS to find distances from source.
///
/// Returns distance array. -1 means unreachable, 0 means self.
fn bfs_distances(
    graph: &petgraph::Graph<crate::Entity, crate::Relation>,
    source: NodeIndex,
    undirected: bool,
) -> Vec<i32> {
    let n = graph.node_count();
    let mut dist = vec![-1_i32; n];
    dist[source.index()] = 0;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        let v_dist = dist[v.index()];

        let neighbors: Vec<NodeIndex> = if undirected {
            crate::algo::unique_neighbors_undirected(graph, v)
        } else {
            crate::algo::unique_neighbors_directed(graph, v, petgraph::Direction::Outgoing)
        };

        for w in neighbors {
            if dist[w.index()] < 0 {
                dist[w.index()] = v_dist + 1;
                queue.push_back(w);
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_closeness_star_directed() {
        let mut kg = KnowledgeGraph::new();
        // Star: Hub -> A, Hub -> B, Hub -> C
        kg.add_triple(Triple::new("Hub", "rel", "A"));
        kg.add_triple(Triple::new("Hub", "rel", "B"));
        kg.add_triple(Triple::new("Hub", "rel", "C"));

        let config = ClosenessConfig::default();
        let scores = closeness_centrality(&kg, config);

        // Hub can reach A, B, C in 1 hop each
        let hub = *scores.get("Hub").unwrap();
        let a = *scores.get("A").unwrap();

        // Hub has positive closeness, leaves have zero (can't reach anyone)
        assert!(hub > 0.0, "Hub should have positive closeness: {hub}");
        assert_eq!(a, 0.0, "A can't reach anyone: {a}");
    }

    #[test]
    fn test_closeness_line() {
        let mut kg = KnowledgeGraph::new();
        // Line: A -> B -> C
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let config = ClosenessConfig {
            normalized: false,
            undirected: true, // treat as undirected
            harmonic: true,
        };
        let scores = closeness_centrality(&kg, config);

        let a = *scores.get("A").unwrap();
        let b = *scores.get("B").unwrap();
        let c = *scores.get("C").unwrap();

        // B is central (dist 1 to A and C), A and C are endpoints
        // Harmonic: B = 1/1 + 1/1 = 2, A = 1/1 + 1/2 = 1.5, C = 1/2 + 1/1 = 1.5
        assert!(b > a, "B should be more central than A: B={b}, A={a}");
        assert!(
            (a - c).abs() < 1e-6,
            "A and C should be symmetric: A={a}, C={c}"
        );
    }

    #[test]
    fn test_closeness_disconnected() {
        let mut kg = KnowledgeGraph::new();
        // Two disconnected edges: A -> B, C -> D
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("C", "rel", "D"));

        let config = ClosenessConfig::default();
        let scores = closeness_centrality(&kg, config);

        // Harmonic centrality handles this gracefully
        // A can only reach B (score > 0 but low)
        let a = *scores.get("A").unwrap();
        assert!(a > 0.0, "A should have some closeness: {a}");
    }

    #[test]
    fn test_closeness_normalized() {
        let mut kg = KnowledgeGraph::new();
        // Complete triangle
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let config = ClosenessConfig {
            normalized: true,
            undirected: false,
            harmonic: true,
        };
        let scores = closeness_centrality(&kg, config);

        // All nodes reach each other in 1 hop
        // Normalized harmonic = (1/1 + 1/1) / 2 = 1.0
        for (name, score) in scores {
            assert!(
                (score - 1.0).abs() < 1e-6,
                "Complete graph should have max closeness: {name}={score}"
            );
        }
    }

    fn assert_score(scores: &HashMap<EntityId, f64>, node: &str, expected: f64) {
        let actual = scores[node];
        assert!(
            (actual - expected).abs() < 1e-12,
            "{node}: expected {expected}, got {actual}"
        );
    }

    fn classic_undirected(kg: &KnowledgeGraph) -> HashMap<EntityId, f64> {
        closeness_centrality(
            kg,
            ClosenessConfig {
                normalized: true,
                undirected: true,
                harmonic: false,
            },
        )
    }

    #[test]
    fn classic_path_star_cycle_and_complete_match_oracles() {
        let mut path = KnowledgeGraph::new();
        path.add_triple(Triple::new("A", "rel", "B"));
        path.add_triple(Triple::new("B", "rel", "C"));
        path.add_triple(Triple::new("C", "rel", "D"));
        let path_scores = classic_undirected(&path);
        for (node, expected) in [("A", 0.5), ("B", 0.75), ("C", 0.75), ("D", 0.5)] {
            assert_score(&path_scores, node, expected);
        }

        let mut star = KnowledgeGraph::new();
        for leaf in ["A", "B", "C"] {
            star.add_triple(Triple::new("Hub", "rel", leaf));
        }
        let star_scores = classic_undirected(&star);
        assert_score(&star_scores, "Hub", 1.0);
        for leaf in ["A", "B", "C"] {
            assert_score(&star_scores, leaf, 3.0 / 5.0);
        }

        let mut cycle = KnowledgeGraph::new();
        for (from, to) in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")] {
            cycle.add_triple(Triple::new(from, "rel", to));
        }
        let cycle_scores = classic_undirected(&cycle);
        for node in ["A", "B", "C", "D"] {
            assert_score(&cycle_scores, node, 3.0 / 4.0);
        }

        let mut complete = KnowledgeGraph::new();
        for (from, to) in [
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
        ] {
            complete.add_triple(Triple::new(from, "rel", to));
        }
        let complete_scores = classic_undirected(&complete);
        for node in ["A", "B", "C", "D"] {
            assert_score(&complete_scores, node, 1.0);
        }
    }

    #[test]
    fn classic_disconnected_scores_use_wasserman_faust_scaling() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("D", "rel", "E"));

        let scores = classic_undirected(&kg);
        for (node, expected) in [
            ("A", 1.0 / 3.0),
            ("B", 0.5),
            ("C", 1.0 / 3.0),
            ("D", 0.25),
            ("E", 0.25),
        ] {
            assert_score(&scores, node, expected);
        }
    }

    #[test]
    fn directed_closeness_explicitly_uses_outward_distances() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let scores = closeness_centrality(
            &kg,
            ClosenessConfig {
                normalized: true,
                undirected: false,
                harmonic: false,
            },
        );

        assert_score(&scores, "A", 2.0 / 3.0);
        assert_score(&scores, "B", 0.5);
        assert_score(&scores, "C", 0.0);
    }
}
