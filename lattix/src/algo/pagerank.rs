//! `PageRank` centrality algorithm.
//!
//! Computes the importance of nodes based on link structure.
//! Higher scores indicate more "important" nodes.
//!
//! The power iteration runs over unique neighbor nodes, so parallel triples
//! do not act as implicit edge weights.

use crate::{EntityId, KnowledgeGraph};
use std::collections::HashMap;

/// Invalid PageRank iteration parameters.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PageRankConfigError {
    /// The damping factor was not finite or outside `[0, 1]`.
    #[error("damping factor must be finite and in [0, 1], got {0}")]
    InvalidDampingFactor(f64),
    /// The convergence tolerance was not finite and strictly positive.
    #[error("tolerance must be finite and greater than zero, got {0}")]
    InvalidTolerance(f64),
}

/// Scores and termination information from a PageRank iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct PageRankResult {
    /// Score for each entity. Scores sum to one for a non-empty graph.
    pub scores: HashMap<EntityId, f64>,
    /// Number of power iterations performed.
    pub iterations: usize,
    /// Whether the L1 score change became smaller than the configured tolerance.
    pub converged: bool,
}

/// `PageRank` configuration.
#[derive(Debug, Clone, Copy)]
pub struct PageRankConfig {
    /// Damping factor (probability of following a link vs teleporting).
    /// Typically 0.85.
    pub damping_factor: f64,
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Convergence tolerance (L1 norm of score changes).
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl PageRankConfig {
    /// Validate the numeric iteration parameters.
    pub fn validate(&self) -> Result<(), PageRankConfigError> {
        if !self.damping_factor.is_finite() || !(0.0..=1.0).contains(&self.damping_factor) {
            return Err(PageRankConfigError::InvalidDampingFactor(
                self.damping_factor,
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(PageRankConfigError::InvalidTolerance(self.tolerance));
        }
        Ok(())
    }
}

/// Compute `PageRank` for all entities.
///
/// Returns a map of `EntityId` -> Score, where scores sum to 1.0.
///
/// # Panics
///
/// Panics if the damping factor or tolerance is invalid. Use [`try_pagerank`]
/// to handle invalid input or inspect convergence.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn pagerank(kg: &KnowledgeGraph, config: PageRankConfig) -> HashMap<EntityId, f64> {
    try_pagerank(kg, config)
        .expect("pagerank requires a finite damping factor in [0, 1] and a positive tolerance")
        .scores
}

/// Checked form of [`pagerank`] with explicit termination information.
#[allow(clippy::cast_precision_loss)]
pub fn try_pagerank(
    kg: &KnowledgeGraph,
    config: PageRankConfig,
) -> Result<PageRankResult, PageRankConfigError> {
    config.validate()?;
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return Ok(PageRankResult {
            scores: HashMap::new(),
            iterations: 0,
            converged: true,
        });
    }

    let adjacency = crate::algo::DedupAdjacency::directed(graph, petgraph::Direction::Outgoing);
    let adjacency = adjacency.rows();

    let n_f = n as f64;
    let damping = config.damping_factor;
    let teleport = (1.0 - damping) / n_f;
    let mut scores = vec![1.0 / n_f; n];
    let mut next = vec![0.0; n];

    let mut iterations = 0;
    let mut converged = false;
    for _ in 0..config.max_iterations {
        iterations += 1;
        next.fill(teleport);

        let dangling: f64 = adjacency
            .iter()
            .enumerate()
            .filter(|(_, neighbors)| neighbors.is_empty())
            .map(|(idx, _)| scores[idx])
            .sum();
        let dangling_share = damping * dangling / n_f;
        for score in &mut next {
            *score += dangling_share;
        }

        for (src, neighbors) in adjacency.iter().enumerate() {
            if neighbors.is_empty() {
                continue;
            }
            let share = damping * scores[src] / neighbors.len() as f64;
            for &dst in neighbors {
                next[dst] += share;
            }
        }

        let diff: f64 = scores
            .iter()
            .zip(next.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();
        std::mem::swap(&mut scores, &mut next);
        if diff < config.tolerance {
            converged = true;
            break;
        }
    }

    let mut result = HashMap::with_capacity(n);
    for (idx, score) in scores.into_iter().enumerate() {
        let entity = &graph[petgraph::graph::NodeIndex::new(idx)];
        result.insert(entity.id.clone(), score);
    }
    Ok(PageRankResult {
        scores: result,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;
    use proptest::prelude::*;

    #[test]
    fn test_pagerank_cycle() {
        let mut kg = KnowledgeGraph::new();
        // A -> B -> C -> A (cycle)
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let scores = pagerank(&kg, PageRankConfig::default());

        // Symmetric cycle: all scores should be equal
        let a = scores.get("A").unwrap();
        let b = scores.get("B").unwrap();
        let c = scores.get("C").unwrap();

        assert!((a - b).abs() < 1e-4, "A={a} B={b}");
        assert!((b - c).abs() < 1e-4, "B={b} C={c}");
        assert!((a - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_pagerank_star() {
        let mut kg = KnowledgeGraph::new();
        // Hub -> A, Hub -> B, Hub -> C (star topology)
        kg.add_triple(Triple::new("Hub", "rel", "A"));
        kg.add_triple(Triple::new("Hub", "rel", "B"));
        kg.add_triple(Triple::new("Hub", "rel", "C"));

        let scores = pagerank(&kg, PageRankConfig::default());

        // Hub has outlinks but no inlinks from the graph
        // A, B, C are dangling (no outlinks)
        let hub = scores.get("Hub").unwrap();
        let a = scores.get("A").unwrap();

        // Dangling nodes receive mass from Hub + teleport
        // Hub only gets teleport (no inlinks)
        assert!(a > hub, "Leaf A ({a}) should rank higher than Hub ({hub})",);
    }

    #[test]
    fn test_pagerank_sums_to_one() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));
        kg.add_triple(Triple::new("A", "rel", "D"));

        let scores = pagerank(&kg, PageRankConfig::default());
        let total: f64 = scores.values().sum();

        assert!(
            (total - 1.0).abs() < 1e-6,
            "Scores should sum to 1.0, got {total}",
        );
    }

    #[test]
    fn checked_pagerank_rejects_invalid_numeric_parameters() {
        let kg = KnowledgeGraph::new();
        for damping in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            let error = try_pagerank(
                &kg,
                PageRankConfig {
                    damping_factor: damping,
                    ..PageRankConfig::default()
                },
            )
            .unwrap_err();
            match error {
                PageRankConfigError::InvalidDampingFactor(value) => {
                    assert_eq!(value.to_bits(), damping.to_bits());
                }
                _ => panic!("expected an invalid damping factor"),
            }
        }
        for tolerance in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let error = try_pagerank(
                &kg,
                PageRankConfig {
                    tolerance,
                    ..PageRankConfig::default()
                },
            )
            .unwrap_err();
            match error {
                PageRankConfigError::InvalidTolerance(value) => {
                    assert_eq!(value.to_bits(), tolerance.to_bits());
                }
                _ => panic!("expected an invalid tolerance"),
            }
        }
    }

    #[test]
    fn checked_pagerank_reports_iteration_limit() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        let result = try_pagerank(
            &kg,
            PageRankConfig {
                max_iterations: 0,
                ..PageRankConfig::default()
            },
        )
        .unwrap();

        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.scores["A"], 0.5);
        assert_eq!(result.scores["B"], 0.5);
    }

    #[test]
    fn checked_pagerank_has_exact_one_step_oracle() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        let result = try_pagerank(
            &kg,
            PageRankConfig {
                damping_factor: 0.5,
                max_iterations: 1,
                tolerance: 1e-12,
            },
        )
        .unwrap();

        assert_eq!(result.iterations, 1);
        assert!(!result.converged);
        assert!((result.scores["A"] - 0.375).abs() < 1e-15);
        assert!((result.scores["B"] - 0.625).abs() < 1e-15);
    }

    proptest! {
        #[test]
        fn pagerank_matches_independent_dense_transition(
            node_count in 2usize..=5,
            edge_bits in proptest::collection::vec(any::<bool>(), 25),
            damping in prop_oneof![Just(0.0), Just(0.25), Just(0.85), Just(1.0)],
            iterations in 0usize..=6,
        ) {
            let requested = &edge_bits[..node_count * node_count];
            let (kg, dense) = crate::algo::test_oracles::graph_with_dense_adjacency(
                node_count,
                requested,
            );
            let uniform = vec![1.0 / node_count as f64; node_count];
            let expected = crate::algo::test_oracles::dense_walk(
                &dense,
                &uniform,
                &uniform,
                damping,
                iterations,
            );
            let actual = try_pagerank(
                &kg,
                PageRankConfig {
                    damping_factor: damping,
                    max_iterations: iterations,
                    tolerance: f64::MIN_POSITIVE,
                },
            ).unwrap();

            for (index, expected_score) in expected.into_iter().enumerate() {
                let id = format!("n{index}");
                prop_assert!(
                    (actual.scores[id.as_str()] - expected_score).abs() < 1e-12,
                    "node={id} actual={} expected={expected_score}",
                    actual.scores[id.as_str()],
                );
            }
        }
    }
}
