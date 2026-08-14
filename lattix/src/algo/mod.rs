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
use petgraph::visit::EdgeRef;

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

/// A compact adjacency view whose rows contain sorted, unique node indices.
///
/// Building this once avoids independently materializing the same graph for
/// algorithms that only need unweighted adjacency.
pub(crate) struct DedupAdjacency(Vec<Vec<usize>>);

impl DedupAdjacency {
    pub(crate) fn directed(
        graph: &petgraph::Graph<crate::Entity, crate::Relation>,
        direction: petgraph::Direction,
    ) -> Self {
        let mut rows = vec![Vec::new(); graph.node_count()];
        for edge in graph.edge_references() {
            let (row, neighbor) = match direction {
                petgraph::Direction::Outgoing => (edge.source(), edge.target()),
                petgraph::Direction::Incoming => (edge.target(), edge.source()),
            };
            rows[row.index()].push(neighbor.index());
        }
        Self::finish(rows)
    }

    pub(crate) fn undirected(graph: &petgraph::Graph<crate::Entity, crate::Relation>) -> Self {
        let mut rows = vec![Vec::new(); graph.node_count()];
        for edge in graph.edge_references() {
            let source = edge.source().index();
            let target = edge.target().index();
            rows[source].push(target);
            rows[target].push(source);
        }
        Self::finish(rows)
    }

    fn finish(mut rows: Vec<Vec<usize>>) -> Self {
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
        }
        Self(rows)
    }

    pub(crate) fn rows(&self) -> &[Vec<usize>] {
        &self.0
    }
}

impl graphops::GraphRef for DedupAdjacency {
    fn node_count(&self) -> usize {
        self.0.len()
    }

    fn neighbors_ref(&self, node: usize) -> &[usize] {
        &self.0[node]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphops::GraphRef;

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

    #[test]
    fn dedup_adjacency_preserves_direction_and_unique_nodes() {
        let mut graph = petgraph::Graph::new();
        let a = graph.add_node(crate::Entity::new("A"));
        let b = graph.add_node(crate::Entity::new("B"));
        let c = graph.add_node(crate::Entity::new("C"));
        graph.add_edge(a, b, crate::Relation::new("first"));
        graph.add_edge(a, b, crate::Relation::new("parallel"));
        graph.add_edge(b, b, crate::Relation::new("loop"));

        let outgoing = DedupAdjacency::directed(&graph, petgraph::Direction::Outgoing);
        assert_eq!(outgoing.node_count(), 3);
        assert_eq!(outgoing.neighbors_ref(a.index()), &[b.index()]);
        assert_eq!(outgoing.neighbors_ref(b.index()), &[b.index()]);
        assert!(outgoing.neighbors_ref(c.index()).is_empty());

        let incoming = DedupAdjacency::directed(&graph, petgraph::Direction::Incoming);
        assert!(incoming.neighbors_ref(a.index()).is_empty());
        assert_eq!(incoming.neighbors_ref(b.index()), &[a.index(), b.index()]);
        assert!(incoming.neighbors_ref(c.index()).is_empty());

        let undirected = DedupAdjacency::undirected(&graph);
        assert_eq!(undirected.neighbors_ref(a.index()), &[b.index()]);
        assert_eq!(undirected.neighbors_ref(b.index()), &[a.index(), b.index()]);
        assert!(undirected.neighbors_ref(c.index()).is_empty());
    }
}

#[cfg(test)]
pub(crate) mod test_oracles {
    use crate::{KnowledgeGraph, Triple};

    pub(crate) fn graph_with_dense_adjacency(
        node_count: usize,
        requested_edges: &[bool],
    ) -> (KnowledgeGraph, Vec<Vec<bool>>) {
        assert!(node_count >= 2);
        assert_eq!(requested_edges.len(), node_count * node_count);
        let mut graph = KnowledgeGraph::new();
        let mut adjacency = vec![vec![false; node_count]; node_count];

        // Materialize every node through a directed star. Its leaves provide
        // dangling-node cases when the generated edges do not add outlinks.
        for (target, edge) in adjacency[0].iter_mut().enumerate().skip(1) {
            graph.add_triple(Triple::new("n0", "base", format!("n{target}")));
            *edge = true;
        }
        for source in 0..node_count {
            for target in 0..node_count {
                if requested_edges[source * node_count + target] {
                    let source_id = format!("n{source}");
                    let target_id = format!("n{target}");
                    graph.add_triple(Triple::new(
                        source_id.as_str(),
                        "generated",
                        target_id.as_str(),
                    ));
                    // Deliberate parallel edge: the dense oracle remains boolean.
                    graph.add_triple(Triple::new(
                        source_id.as_str(),
                        "parallel",
                        target_id.as_str(),
                    ));
                    adjacency[source][target] = true;
                }
            }
        }
        (graph, adjacency)
    }

    pub(crate) fn dense_walk(
        adjacency: &[Vec<bool>],
        initial: &[f64],
        personalization: &[f64],
        damping: f64,
        iterations: usize,
    ) -> Vec<f64> {
        let n = adjacency.len();
        let mut scores = initial.to_vec();
        for _ in 0..iterations {
            let mut next: Vec<_> = personalization
                .iter()
                .map(|value| (1.0 - damping) * value)
                .collect();
            for source in 0..n {
                let degree = adjacency[source].iter().filter(|&&edge| edge).count();
                if degree == 0 {
                    for (value, &personalized) in next.iter_mut().zip(personalization) {
                        *value += damping * scores[source] * personalized;
                    }
                } else {
                    let share = damping * scores[source] / degree as f64;
                    for (target, &edge) in adjacency[source].iter().enumerate() {
                        if edge {
                            next[target] += share;
                        }
                    }
                }
            }
            scores = next;
        }
        scores
    }
}
