//! Algorithms for graph analysis and embedding generation.
//!
//! This module contains implementations of graph algorithms:
//!
//! - **Centrality**: Measure node importance ([`centrality`])
//! - **Random walks**: Node2Vec-style biased walks ([`random_walk`])
//! - **Components**: Find connected components ([`components`])
//! - **Sampling**: Mini-batch sampling for GNNs ([`sampling`])
//! - **PPR**: Personalized PageRank from a seed entity ([`ppr`])
//! - **Label propagation**: Community detection ([`label_propagation`])
//!
//! # Centrality Overview
//!
//! | Algorithm | Question | Complexity |
//! |-----------|----------|------------|
//! These complexities include the cost of deduplicating parallel triples into
//! unique neighbor nodes, where `d_max` is the maximum stored degree.
//!
//! | Algorithm | Question | Complexity |
//! |-----------|----------|------------|
//! | Degree | How many connections? | O(V + E log d_max) |
//! | Betweenness | Bridge between communities? | O(VE log d_max) |
//! | Closeness | How close to everyone? | O(VE log d_max) |
//! | Eigenvector | Connected to important nodes? | O(E log d_max × iter) |
//! | Katz | Reachable via damped paths? | O(E log d_max × iter) |
//! | PageRank | Random walk equilibrium? | O(E log d_max + E × iter) |
//! | HITS | Hub or authority? | O(E log d_max × iter) |

/// Centrality algorithms for measuring node importance.
pub mod centrality;

/// Random walk algorithm (Node2Vec style).
pub mod random_walk;

/// PageRank centrality algorithm (also available via [`centrality`]).
pub mod pagerank;

/// Connected components algorithm.
pub mod components;

/// Graph sampling algorithms (e.g. for GNNs).
pub mod sampling;

/// Personalized PageRank from a seed entity.
pub mod ppr;

/// Label propagation community detection.
pub mod label_propagation;

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::EntityId;

/// Return the top-n scored entities, sorted descending by score.
///
/// Finite scores and infinities use [`f64::total_cmp`]; NaN scores sort last.
/// Equal scores are ordered by entity ID, so the result does not depend on
/// [`HashMap`] insertion or iteration order.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use lattix::EntityId;
/// use lattix::algo::top_n;
///
/// let scores: HashMap<EntityId, f64> = [
///     (EntityId::from("A"), 0.5),
///     (EntityId::from("B"), 0.3),
///     (EntityId::from("C"), 0.2),
/// ].into_iter().collect();
///
/// let top = top_n(&scores, 2);
/// assert_eq!(top.len(), 2);
/// assert_eq!(top[0].0.as_str(), "A");
/// assert_eq!(top[1].0.as_str(), "B");
/// ```
#[must_use]
pub fn top_n(scores: &HashMap<EntityId, f64>, n: usize) -> Vec<(EntityId, f64)> {
    let mut entries: Vec<(EntityId, f64)> = scores.iter().map(|(k, &v)| (k.clone(), v)).collect();
    entries.sort_by(|a, b| {
        a.1.is_nan()
            .cmp(&b.1.is_nan())
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| a.0.as_str().cmp(b.0.as_str()))
    });
    entries.truncate(n);
    entries
}

pub(crate) fn unique_neighbors_directed(
    graph: &petgraph::Graph<crate::Entity, crate::Relation>,
    node: NodeIndex,
    direction: petgraph::Direction,
) -> Vec<NodeIndex> {
    let mut neighbors: Vec<_> = graph.neighbors_directed(node, direction).collect();
    dedup_nodes(&mut neighbors);
    neighbors
}

pub(crate) fn unique_neighbors_undirected(
    graph: &petgraph::Graph<crate::Entity, crate::Relation>,
    node: NodeIndex,
) -> Vec<NodeIndex> {
    let mut neighbors: Vec<_> = graph.neighbors_undirected(node).collect();
    dedup_nodes(&mut neighbors);
    neighbors
}

fn dedup_nodes(nodes: &mut Vec<NodeIndex>) {
    nodes.sort_unstable_by_key(|idx| idx.index());
    nodes.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_n_breaks_equal_score_ties_by_entity_id() {
        let permutations = [
            [("z", 0.5), ("a", 0.5), ("m", 0.5)],
            [("m", 0.5), ("z", 0.5), ("a", 0.5)],
            [("a", 0.5), ("m", 0.5), ("z", 0.5)],
        ];

        for input in permutations {
            let scores: HashMap<_, _> = input
                .into_iter()
                .map(|(id, score)| (EntityId::from(id), score))
                .collect();
            let ids: Vec<_> = top_n(&scores, 3)
                .into_iter()
                .map(|(id, _)| id.into_string())
                .collect();
            assert_eq!(ids, ["a", "m", "z"]);
        }
    }

    #[test]
    fn top_n_uses_a_total_score_order() {
        let scores = HashMap::from([
            (EntityId::from("negative_infinity"), f64::NEG_INFINITY),
            (EntityId::from("negative_zero"), -0.0),
            (EntityId::from("positive_zero"), 0.0),
            (EntityId::from("infinity"), f64::INFINITY),
            (EntityId::from("nan"), f64::NAN),
        ]);

        let ids: Vec<_> = top_n(&scores, scores.len())
            .into_iter()
            .map(|(id, _)| id.into_string())
            .collect();
        assert_eq!(
            ids,
            [
                "infinity",
                "positive_zero",
                "negative_zero",
                "negative_infinity",
                "nan"
            ]
        );
    }
}
