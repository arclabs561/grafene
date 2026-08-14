//! Personalized PageRank (PPR).
//!
//! Computes node importance relative to a specific seed entity,
//! measuring proximity in the graph's link structure.
//!
//! The power iteration runs over unique neighbor nodes, so parallel triples
//! do not act as implicit edge weights.

use crate::{EntityId, KnowledgeGraph};
use std::collections::HashMap;

/// Invalid input to personalized PageRank.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PprError {
    /// The damping factor was not finite or outside `[0, 1]`.
    #[error("damping factor must be finite and in [0, 1], got {0}")]
    InvalidDamping(f64),
    /// The convergence tolerance was not finite and strictly positive.
    #[error("tolerance must be finite and greater than zero, got {0}")]
    InvalidTolerance(f64),
    /// The requested seed is absent from the graph.
    #[error("seed entity is not present in the graph")]
    SeedNotFound,
}

/// Scores and termination information from a personalized PageRank iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct PprResult {
    /// Score for each entity. Scores sum to one for a non-empty graph.
    pub scores: HashMap<EntityId, f64>,
    /// Number of power iterations performed.
    pub iterations: usize,
    /// Whether the L1 score change became smaller than the configured tolerance.
    pub converged: bool,
}

/// PPR configuration.
#[derive(Debug, Clone, Copy)]
pub struct PprConfig {
    /// Damping factor (probability of following a link vs teleporting back to seed).
    /// Typically 0.85.
    pub damping: f64,
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Convergence tolerance (L1 norm of score changes).
    pub tolerance: f64,
}

impl Default for PprConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl PprConfig {
    /// Validate the numeric iteration parameters.
    pub fn validate(&self) -> Result<(), PprError> {
        if !self.damping.is_finite() || !(0.0..=1.0).contains(&self.damping) {
            return Err(PprError::InvalidDamping(self.damping));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(PprError::InvalidTolerance(self.tolerance));
        }
        Ok(())
    }
}

/// Compute personalized PageRank from a seed entity.
///
/// Returns scores keyed by entity ID. Higher scores indicate
/// entities closer/more connected to the seed in the graph's link structure.
///
/// Returns an empty map if the graph is empty or the seed entity is not found.
/// Invalid numeric parameters panic; use [`try_personalized_pagerank`] to
/// handle them or inspect convergence.
///
/// # Example
///
/// ```
/// use lattix::{KnowledgeGraph, Triple};
/// use lattix::algo::ppr::{personalized_pagerank, PprConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// kg.add_triple(Triple::new("Alice", "knows", "Bob"));
/// kg.add_triple(Triple::new("Bob", "knows", "Carol"));
///
/// let scores = personalized_pagerank(&kg, "Alice", PprConfig::default());
/// assert!(scores.contains_key("Alice"));
/// assert!(scores.contains_key("Bob"));
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn personalized_pagerank(
    kg: &KnowledgeGraph,
    seed: &str,
    config: PprConfig,
) -> HashMap<EntityId, f64> {
    match try_personalized_pagerank(kg, seed, config) {
        Ok(result) => result.scores,
        Err(PprError::SeedNotFound) => HashMap::new(),
        Err(error) => panic!("invalid personalized PageRank configuration: {error}"),
    }
}

/// Checked form of [`personalized_pagerank`] with explicit termination information.
#[allow(clippy::cast_precision_loss)]
pub fn try_personalized_pagerank(
    kg: &KnowledgeGraph,
    seed: &str,
    config: PprConfig,
) -> Result<PprResult, PprError> {
    config.validate()?;
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return Ok(PprResult {
            scores: HashMap::new(),
            iterations: 0,
            converged: true,
        });
    }

    // Find the seed node index
    let seed_id = crate::EntityId::from(seed);
    let seed_idx = match kg.get_node_index(&seed_id) {
        Some(idx) => idx.index(),
        None => return Err(PprError::SeedNotFound),
    };

    let mut personalization = vec![0.0; n];
    personalization[seed_idx] = 1.0;

    let adjacency = crate::algo::DedupAdjacency::directed(graph, petgraph::Direction::Outgoing);
    let adjacency = adjacency.rows();

    let mut scores = personalization.clone();
    let mut next = vec![0.0; n];

    let mut iterations = 0;
    let mut converged = false;
    for _ in 0..config.max_iterations {
        iterations += 1;
        for (idx, value) in next.iter_mut().enumerate() {
            *value = (1.0 - config.damping) * personalization[idx];
        }

        let dangling: f64 = adjacency
            .iter()
            .enumerate()
            .filter(|(_, neighbors)| neighbors.is_empty())
            .map(|(idx, _)| scores[idx])
            .sum();
        let dangling_share = config.damping * dangling;
        for (idx, value) in next.iter_mut().enumerate() {
            *value += dangling_share * personalization[idx];
        }

        for (src, neighbors) in adjacency.iter().enumerate() {
            if neighbors.is_empty() {
                continue;
            }
            let share = config.damping * scores[src] / neighbors.len() as f64;
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
    Ok(PprResult {
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
    fn ppr_seed_scores_highest() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let scores = personalized_pagerank(&kg, "A", PprConfig::default());

        let a = *scores.get("A").unwrap();
        let b = *scores.get("B").unwrap();
        let c = *scores.get("C").unwrap();

        // Seed should have the highest score in PPR
        assert!(a > b, "Seed A ({a}) should score higher than B ({b})");
        assert!(a > c, "Seed A ({a}) should score higher than C ({c})");
    }

    #[test]
    fn ppr_missing_seed_returns_empty() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        let scores = personalized_pagerank(&kg, "Z", PprConfig::default());
        assert!(scores.is_empty());
    }

    #[test]
    fn ppr_empty_graph() {
        let kg = KnowledgeGraph::new();
        let scores = personalized_pagerank(&kg, "A", PprConfig::default());
        assert!(scores.is_empty());
    }

    #[test]
    fn ppr_scores_sum_to_one() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));
        kg.add_triple(Triple::new("A", "rel", "D"));

        let scores = personalized_pagerank(&kg, "A", PprConfig::default());
        let total: f64 = scores.values().sum();

        assert!(
            (total - 1.0).abs() < 1e-6,
            "Scores should sum to 1.0, got {total}",
        );
    }

    #[test]
    fn checked_ppr_rejects_invalid_input() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        for damping in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            let error = try_personalized_pagerank(
                &kg,
                "A",
                PprConfig {
                    damping,
                    ..PprConfig::default()
                },
            )
            .unwrap_err();
            match error {
                PprError::InvalidDamping(value) => {
                    assert_eq!(value.to_bits(), damping.to_bits());
                }
                _ => panic!("expected an invalid damping factor"),
            }
        }
        for tolerance in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let error = try_personalized_pagerank(
                &kg,
                "A",
                PprConfig {
                    tolerance,
                    ..PprConfig::default()
                },
            )
            .unwrap_err();
            match error {
                PprError::InvalidTolerance(value) => {
                    assert_eq!(value.to_bits(), tolerance.to_bits());
                }
                _ => panic!("expected an invalid tolerance"),
            }
        }
        assert_eq!(
            try_personalized_pagerank(&kg, "missing", PprConfig::default()).unwrap_err(),
            PprError::SeedNotFound
        );
    }

    #[test]
    fn checked_ppr_reports_iteration_limit() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        let result = try_personalized_pagerank(
            &kg,
            "A",
            PprConfig {
                max_iterations: 0,
                ..PprConfig::default()
            },
        )
        .unwrap();

        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.scores["A"], 1.0);
        assert_eq!(result.scores["B"], 0.0);
    }

    #[test]
    fn checked_ppr_has_exact_one_step_oracle() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));

        let result = try_personalized_pagerank(
            &kg,
            "A",
            PprConfig {
                damping: 0.5,
                max_iterations: 1,
                tolerance: 1e-12,
            },
        )
        .unwrap();

        assert_eq!(result.iterations, 1);
        assert!(!result.converged);
        assert!((result.scores["A"] - 0.5).abs() < 1e-15);
        assert!((result.scores["B"] - 0.5).abs() < 1e-15);
    }

    proptest! {
        #[test]
        fn one_hot_ppr_matches_independent_dense_transition(
            node_count in 2usize..=5,
            edge_bits in proptest::collection::vec(any::<bool>(), 25),
            seed in 0usize..5,
            damping in prop_oneof![Just(0.0), Just(0.25), Just(0.85), Just(1.0)],
            iterations in 0usize..=6,
        ) {
            let seed = seed % node_count;
            let requested = &edge_bits[..node_count * node_count];
            let (kg, dense) = crate::algo::test_oracles::graph_with_dense_adjacency(
                node_count,
                requested,
            );
            let mut one_hot = vec![0.0; node_count];
            one_hot[seed] = 1.0;
            let expected = crate::algo::test_oracles::dense_walk(
                &dense,
                &one_hot,
                &one_hot,
                damping,
                iterations,
            );
            let seed_id = format!("n{seed}");
            let actual = try_personalized_pagerank(
                &kg,
                &seed_id,
                PprConfig {
                    damping,
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
