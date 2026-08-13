//! `Node2Vec`-style random walk generation.
//!
//! Implements biased 2nd-order random walks as described in:
//! Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks" (KDD 2016)
//!
//! ## Performance notes
//!
//! - Samples from normalized transition weights in O(d) time per step
//! - Caches previous node's neighbors in `HashSet` for O(1) membership test
//! - Parallelized across walk iterations via rayon

use crate::{Error, KnowledgeGraph};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;

/// Invalid node2vec bias parameters.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RandomWalkConfigError {
    /// The return parameter `p` must be finite and greater than zero.
    #[error("return parameter p must be finite and greater than zero, got {0}")]
    InvalidP(f32),
    /// The in-out parameter `q` must be finite and greater than zero.
    #[error("in-out parameter q must be finite and greater than zero, got {0}")]
    InvalidQ(f32),
}

/// Configuration for random walks.
#[derive(Debug, Clone, Copy)]
pub struct RandomWalkConfig {
    /// Length of each random walk.
    pub walk_length: usize,
    /// Number of walks to start from each node.
    pub num_walks: usize,
    /// Return parameter (p) - likelihood of returning to previous node.
    /// - p > 1: less likely to backtrack
    /// - p < 1: more likely to backtrack
    pub p: f32,
    /// In-out parameter (q) - controls BFS vs DFS behavior.
    /// - q > 1: BFS-like (local exploration)
    /// - q < 1: DFS-like (outward exploration)
    pub q: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RandomWalkConfig {
    fn default() -> Self {
        Self {
            walk_length: 80,
            num_walks: 10,
            p: 1.0,
            q: 1.0,
            seed: 42,
        }
    }
}

impl RandomWalkConfig {
    /// Validate the node2vec bias parameters.
    ///
    /// Both `p` and `q` must be finite and strictly positive. Walk length and
    /// count are not bias parameters and may be zero.
    pub fn validate(&self) -> Result<(), RandomWalkConfigError> {
        if !self.p.is_finite() || self.p <= 0.0 {
            return Err(RandomWalkConfigError::InvalidP(self.p));
        }
        if !self.q.is_finite() || self.q <= 0.0 {
            return Err(RandomWalkConfigError::InvalidQ(self.q));
        }
        Ok(())
    }
}

/// Generate random walks for all nodes in the graph.
///
/// # Arguments
/// * `kg` - The Knowledge Graph
/// * `config` - Walk configuration
///
/// # Returns
/// A vector of walks, where each walk is a vector of entity IDs.
#[must_use]
pub fn generate_walks(kg: &KnowledgeGraph, config: RandomWalkConfig) -> Vec<Vec<String>> {
    let walker = Node2Vec::new(kg, config);
    walker.walk()
}

/// Generate random walks after validating the node2vec bias parameters.
pub fn try_generate_walks(
    kg: &KnowledgeGraph,
    config: RandomWalkConfig,
) -> Result<Vec<Vec<String>>, RandomWalkConfigError> {
    Ok(Node2Vec::try_new(kg, config)?.walk())
}

/// A walk corpus over dense node indices (0..N).
///
/// This is the format expected by Node2Vec/SkipGram pipelines, where embeddings are stored as
/// a flat matrix indexed by node id.
///
/// Invariants:
/// - `node_ids.len() == N`
/// - each entry in `walks` is a sequence of integers in `[0, N)`
#[derive(Debug, Clone)]
pub struct WalkCorpus {
    /// Dense node index -> graph node identifier.
    pub node_ids: Vec<String>,
    /// Random walks as dense node indices.
    pub walks: Vec<Vec<u32>>,
}

/// Generate random walks plus a stable dense node-id mapping.
///
/// This is a compatibility helper for higher layers that need dense indices.
///
/// Note: the underlying walk generator operates on `KnowledgeGraph`'s internal `petgraph`
/// structure. We reuse `generate_walks` (which yields entity IDs) and then map entity IDs to
/// dense indices. This avoids duplicating the node2vec walk logic.
pub fn generate_walk_corpus(
    kg: &KnowledgeGraph,
    config: RandomWalkConfig,
) -> crate::Result<WalkCorpus> {
    // Build a stable dense ordering over node identifiers.
    let graph = kg.as_petgraph();
    let mut node_indices: Vec<_> = graph.node_indices().collect();
    node_indices.sort_by_key(|n| n.index());

    let node_ids: Vec<String> = node_indices
        .iter()
        .map(|&n| graph[n].id.as_str().to_owned())
        .collect();

    let id_to_dense: HashMap<String, u32> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i as u32))
        .collect();

    let walks_ids = generate_walks(kg, config);
    let mut walks = Vec::with_capacity(walks_ids.len());
    for w in walks_ids {
        let mut dense_walk = Vec::with_capacity(w.len());
        for id in w {
            let idx = *id_to_dense.get(&id).ok_or(Error::EntityNotFound(id))?;
            dense_walk.push(idx);
        }
        walks.push(dense_walk);
    }

    Ok(WalkCorpus { node_ids, walks })
}

/// `Node2Vec` random walker.
pub struct Node2Vec<'a> {
    kg: &'a KnowledgeGraph,
    config: RandomWalkConfig,
}

impl<'a> Node2Vec<'a> {
    /// Create a new `Node2Vec` walker.
    ///
    /// This preserves the original infallible constructor. Invalid bias
    /// parameters fail immediately when [`Node2Vec::walk`] is called. Use
    /// [`Node2Vec::try_new`] to validate them at construction time.
    #[must_use]
    pub const fn new(kg: &'a KnowledgeGraph, config: RandomWalkConfig) -> Self {
        Self { kg, config }
    }

    /// Create a walker after validating its node2vec bias parameters.
    pub fn try_new(
        kg: &'a KnowledgeGraph,
        config: RandomWalkConfig,
    ) -> Result<Self, RandomWalkConfigError> {
        config.validate()?;
        Ok(Self { kg, config })
    }

    /// Generate all random walks using parallel processing.
    ///
    /// # Panics
    ///
    /// Panics if `p` or `q` is zero, negative, or non-finite. Use
    /// [`Node2Vec::try_new`] to validate configuration at construction time.
    #[must_use]
    pub fn walk(&self) -> Vec<Vec<String>> {
        self.config
            .validate()
            .expect("invalid node2vec random-walk configuration");
        let node_indices: Vec<_> = self.kg.as_petgraph().node_indices().collect();
        let is_unbiased = (self.config.p - 1.0).abs() < f32::EPSILON
            && (self.config.q - 1.0).abs() < f32::EPSILON;

        (0..self.config.num_walks)
            .into_par_iter()
            .flat_map(|iter_idx| {
                let mut rng = XorShiftRng::seed_from_u64(self.config.seed + iter_idx as u64);
                let mut walks = Vec::with_capacity(node_indices.len());

                // Shuffle start nodes to avoid bias
                let mut shuffled = node_indices.clone();
                shuffled.shuffle(&mut rng);

                for &start in &shuffled {
                    let walk = if is_unbiased {
                        self.unbiased_walk(start, &mut rng)
                    } else {
                        self.biased_walk(start, &mut rng)
                    };
                    walks.push(walk);
                }
                walks
            })
            .collect()
    }

    /// Uniform random walk (`DeepWalk`) - O(1) per step.
    fn unbiased_walk<R: Rng>(&self, start: petgraph::graph::NodeIndex, rng: &mut R) -> Vec<String> {
        let graph = self.kg.as_petgraph();
        let mut walk = Vec::with_capacity(self.config.walk_length);
        walk.push(graph[start].id.as_str().to_owned());

        let mut curr = start;
        for _ in 1..self.config.walk_length {
            let neighbors: Vec<_> = graph.neighbors(curr).collect();
            if neighbors.is_empty() {
                break;
            }
            curr = *neighbors
                .choose(rng)
                .expect("internal: neighbors non-empty after check");
            walk.push(graph[curr].id.as_str().to_owned());
        }
        walk
    }

    /// Biased 2nd-order random walk.
    fn biased_walk<R: Rng>(&self, start: petgraph::graph::NodeIndex, rng: &mut R) -> Vec<String> {
        let graph = self.kg.as_petgraph();
        let mut walk = Vec::with_capacity(self.config.walk_length);
        walk.push(graph[start].id.as_str().to_owned());

        let mut curr = start;
        let mut prev: Option<petgraph::graph::NodeIndex> = None;
        let mut prev_neighbors: HashSet<petgraph::graph::NodeIndex> = HashSet::new();

        for _ in 1..self.config.walk_length {
            let neighbors: Vec<_> = graph.neighbors(curr).collect();
            if neighbors.is_empty() {
                break;
            }

            let next = if let Some(prev_node) = prev {
                self.sample_biased(rng, prev_node, &prev_neighbors, &neighbors)
            } else {
                // First step: uniform
                *neighbors
                    .choose(rng)
                    .expect("internal: neighbors non-empty after check")
            };

            walk.push(graph[next].id.as_str().to_owned());

            // Update state: cache current's neighbors as they become "prev_neighbors"
            prev = Some(curr);
            prev_neighbors.clear();
            prev_neighbors.extend(graph.neighbors(curr));
            curr = next;
        }
        walk
    }

    /// Sample the next node from the normalized node2vec transition weights.
    ///
    /// Scaling by the largest weight among the candidates prevents overflow
    /// for extreme finite `p` or `q`. The single bounded pass also avoids the
    /// unbounded rejection time that occurs when a large-weight transition
    /// class is absent from `neighbors`.
    fn sample_biased<R: Rng>(
        &self,
        rng: &mut R,
        prev_node: petgraph::graph::NodeIndex,
        prev_neighbors: &HashSet<petgraph::graph::NodeIndex>,
        neighbors: &[petgraph::graph::NodeIndex],
    ) -> petgraph::graph::NodeIndex {
        let p = f64::from(self.config.p);
        let q = f64::from(self.config.q);

        let transition_weight = |candidate| {
            if candidate == prev_node {
                1.0 / p // Backtrack
            } else if prev_neighbors.contains(&candidate) {
                1.0 // Triangle (dist=1 from prev)
            } else {
                1.0 / q // Move away (dist=2 from prev)
            }
        };

        let max_weight = neighbors
            .iter()
            .copied()
            .map(transition_weight)
            .fold(0.0_f64, f64::max);
        let total: f64 = neighbors
            .iter()
            .copied()
            .map(|candidate| transition_weight(candidate) / max_weight)
            .sum();
        let mut draw = rng.random::<f64>() * total;

        for &candidate in neighbors {
            let weight = transition_weight(candidate) / max_weight;
            if draw < weight {
                return candidate;
            }
            draw -= weight;
        }

        *neighbors
            .last()
            .expect("internal: neighbors non-empty (caller guarantees)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn test_random_walk_uniform() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));

        let config = RandomWalkConfig {
            walk_length: 10,
            num_walks: 2,
            p: 1.0,
            q: 1.0,
            seed: 42,
        };

        let walks = generate_walks(&kg, config);
        assert_eq!(walks.len(), 3 * 2); // 3 nodes * 2 walks
        for walk in &walks {
            assert!(!walk.is_empty());
        }
    }

    #[test]
    fn test_random_walk_biased() {
        let mut kg = KnowledgeGraph::new();
        // Create a small graph: A <-> B <-> C <-> D
        for (a, b) in [("A", "B"), ("B", "C"), ("C", "D")] {
            kg.add_triple(Triple::new(a, "rel", b));
            kg.add_triple(Triple::new(b, "rel", a));
        }

        let config = RandomWalkConfig {
            walk_length: 20,
            num_walks: 5,
            p: 0.5, // More likely to backtrack
            q: 2.0, // Less likely to explore
            seed: 123,
        };

        let walks = generate_walks(&kg, config);
        assert_eq!(walks.len(), 4 * 5); // 4 nodes * 5 walks

        // With p=0.5 (backtrack likely) and q=2.0 (outward unlikely),
        // walks should tend to stay local
        for walk in &walks {
            assert!(walk.len() > 1);
        }
    }

    #[test]
    fn test_random_walk_reproducible() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));

        let config = RandomWalkConfig {
            walk_length: 10,
            num_walks: 3,
            seed: 999,
            ..Default::default()
        };

        let walks1 = generate_walks(&kg, config);
        let walks2 = generate_walks(&kg, config);

        // Same seed should produce same walks
        assert_eq!(walks1, walks2);
    }

    #[test]
    fn config_rejects_non_positive_and_non_finite_biases() {
        for p in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let config = RandomWalkConfig {
                p,
                ..Default::default()
            };
            assert!(matches!(
                config.validate(),
                Err(RandomWalkConfigError::InvalidP(value)) if value.to_bits() == p.to_bits()
            ));
        }

        for q in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let config = RandomWalkConfig {
                q,
                ..Default::default()
            };
            assert!(matches!(
                config.validate(),
                Err(RandomWalkConfigError::InvalidQ(value)) if value.to_bits() == q.to_bits()
            ));
        }
    }

    #[test]
    fn checked_constructor_accepts_extreme_finite_biases() {
        let kg = KnowledgeGraph::new();
        for (p, q) in [(f32::MIN_POSITIVE, f32::MAX), (f32::MAX, f32::MIN_POSITIVE)] {
            let config = RandomWalkConfig {
                p,
                q,
                ..Default::default()
            };
            assert!(Node2Vec::try_new(&kg, config).is_ok());
        }
    }

    #[test]
    #[should_panic(expected = "invalid node2vec random-walk configuration")]
    fn infallible_constructor_fails_fast_before_walking_with_invalid_bias() {
        let kg = KnowledgeGraph::new();
        let walker = Node2Vec::new(
            &kg,
            RandomWalkConfig {
                p: 0.0,
                ..Default::default()
            },
        );

        let _ = walker.walk();
    }

    #[test]
    fn biased_sampler_matches_explicit_normalized_distribution() {
        use petgraph::graph::NodeIndex;

        let kg = KnowledgeGraph::new();
        let walker = Node2Vec::try_new(
            &kg,
            RandomWalkConfig {
                p: 2.0,
                q: 4.0,
                ..Default::default()
            },
        )
        .unwrap();
        let previous = NodeIndex::new(0);
        let triangle = NodeIndex::new(1);
        let outward = NodeIndex::new(2);
        let neighbors = [previous, triangle, outward];
        let previous_neighbors = HashSet::from([triangle]);
        let expected_weights = [0.5_f64, 1.0, 0.25];
        let expected_total: f64 = expected_weights.iter().sum();
        let expected = expected_weights.map(|weight| weight / expected_total);

        let mut rng = XorShiftRng::seed_from_u64(73);
        let mut counts = [0_usize; 3];
        const DRAWS: usize = 200_000;
        for _ in 0..DRAWS {
            let sampled = walker.sample_biased(&mut rng, previous, &previous_neighbors, &neighbors);
            counts[sampled.index()] += 1;
        }

        for (count, expected_probability) in counts.into_iter().zip(expected) {
            let observed = count as f64 / DRAWS as f64;
            assert!(
                (observed - expected_probability).abs() < 0.005,
                "observed {observed}, expected {expected_probability}"
            );
        }
    }

    #[test]
    fn absent_extreme_weight_class_does_not_stall_sampling() {
        use petgraph::graph::NodeIndex;

        let kg = KnowledgeGraph::new();
        let walker = Node2Vec::try_new(
            &kg,
            RandomWalkConfig {
                p: f32::MIN_POSITIVE,
                q: f32::MAX,
                ..Default::default()
            },
        )
        .unwrap();
        let previous = NodeIndex::new(0);
        let only_triangle = NodeIndex::new(1);
        let neighbors = [only_triangle];
        let previous_neighbors = HashSet::from([only_triangle]);
        let mut rng = XorShiftRng::seed_from_u64(11);

        assert_eq!(
            walker.sample_biased(&mut rng, previous, &previous_neighbors, &neighbors),
            only_triangle
        );
    }
}
