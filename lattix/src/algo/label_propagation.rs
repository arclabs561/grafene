//! Label propagation community detection.
//!
//! Detects communities by iteratively assigning each node the most frequent
//! label among its neighbors (Raghavan et al., 2007).
//!
//! The core algorithm is provided by [`graphops`]; this module wraps it
//! with a `KnowledgeGraph`-specific API that maps petgraph node indices
//! back to entity IDs.

use crate::KnowledgeGraph;
use std::collections::HashMap;

/// Label propagation configuration.
#[derive(Debug, Clone, Copy)]
pub struct LabelPropagationConfig {
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Seed for deterministic tie-breaking.
    pub seed: u64,
}

impl Default for LabelPropagationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            seed: 42,
        }
    }
}

/// Detect communities via label propagation.
///
/// Returns a map from entity ID to community label (`usize`).
/// Nodes in the same community share the same label. Labels are
/// contiguous in `0..k`.
///
/// The algorithm treats the directed graph as undirected (both
/// in- and out-neighbors are considered).
///
/// # Example
///
/// ```
/// use lattix::{KnowledgeGraph, Triple};
/// use lattix::algo::label_propagation::{label_propagation, LabelPropagationConfig};
///
/// let mut kg = KnowledgeGraph::new();
/// // Two loosely connected clusters
/// kg.add_triple(Triple::new("A", "rel", "B"));
/// kg.add_triple(Triple::new("B", "rel", "A"));
/// kg.add_triple(Triple::new("C", "rel", "D"));
/// kg.add_triple(Triple::new("D", "rel", "C"));
/// kg.add_triple(Triple::new("B", "bridge", "C"));
///
/// let communities = label_propagation(&kg, LabelPropagationConfig::default());
/// assert_eq!(communities.len(), 4);
/// ```
#[must_use]
pub fn label_propagation(
    kg: &KnowledgeGraph,
    config: LabelPropagationConfig,
) -> HashMap<String, usize> {
    let graph = kg.as_petgraph();
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    // Label propagation is defined on undirected graphs, so the shared view
    // includes both in- and out-neighbors.
    let adapter = crate::algo::DedupAdjacency::undirected(graph);
    let labels =
        graphops::partition::label_propagation(&adapter, config.max_iterations, config.seed);

    // Map indices back to entity IDs
    let mut result = HashMap::with_capacity(n);
    for (idx, label) in labels.into_iter().enumerate() {
        let entity = &graph[petgraph::graph::NodeIndex::new(idx)];
        result.insert(entity.id.as_str().to_owned(), label);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    #[test]
    fn label_propagation_finds_communities() {
        let mut kg = KnowledgeGraph::new();
        // Cluster 1: A-B-C fully connected
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "A"));
        kg.add_triple(Triple::new("A", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "B"));
        // Cluster 2: D-E-F fully connected
        kg.add_triple(Triple::new("D", "rel", "E"));
        kg.add_triple(Triple::new("E", "rel", "D"));
        kg.add_triple(Triple::new("D", "rel", "F"));
        kg.add_triple(Triple::new("F", "rel", "D"));
        kg.add_triple(Triple::new("E", "rel", "F"));
        kg.add_triple(Triple::new("F", "rel", "E"));
        // Weak bridge
        kg.add_triple(Triple::new("C", "bridge", "D"));

        let communities = label_propagation(&kg, LabelPropagationConfig::default());
        assert_eq!(communities.len(), 6);

        // Nodes within each cluster should share a label
        let a = communities["A"];
        let b = communities["B"];
        let c = communities["C"];
        let d = communities["D"];
        let e = communities["E"];
        let f = communities["F"];

        assert_eq!(a, b, "A and B should be in the same community");
        assert_eq!(a, c, "A and C should be in the same community");
        assert_eq!(d, e, "D and E should be in the same community");
        assert_eq!(d, f, "D and F should be in the same community");
    }

    #[test]
    fn label_propagation_empty_graph() {
        let kg = KnowledgeGraph::new();
        let communities = label_propagation(&kg, LabelPropagationConfig::default());
        assert!(communities.is_empty());
    }

    #[test]
    fn label_propagation_deterministic() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(Triple::new("A", "rel", "B"));
        kg.add_triple(Triple::new("B", "rel", "C"));
        kg.add_triple(Triple::new("C", "rel", "A"));

        let config = LabelPropagationConfig::default();
        let a = label_propagation(&kg, config);
        let b = label_propagation(&kg, config);
        assert_eq!(a, b);
    }
}
