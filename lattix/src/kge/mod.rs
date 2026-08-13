//! Benchmark data and evaluation for knowledge graph embedding experiments.
//!
//! This module provides the data layer shared by KGE training and evaluation
//! pipelines: dataset loading, string interning, and rank-based metrics.
//!
//! # Dataset format
//!
//! Standard KGE benchmarks (WN18RR, FB15k-237, YAGO3-10) use a directory with
//! three TSV files:
//!
//! ```text
//! data/wn18rr/
//!   train.txt    # head\trelation\ttail per line
//!   valid.txt
//!   test.txt
//! ```
//!
//! [`load_dataset`](crate::kge::load_dataset) reads this layout into a
//! [`Dataset`](crate::kge::Dataset) of string triples.
//! [`Dataset::into_interned`](crate::kge::Dataset::into_interned) maps strings
//! to integer IDs for training.
//!
//! # Metrics
//!
//! Rank-based metrics for filtered link prediction evaluation:
//!
//! | Metric | Function | Direction |
//! |--------|----------|-----------|
//! | Mean Reciprocal Rank | [`mean_reciprocal_rank`](crate::kge::mean_reciprocal_rank) | Higher is better |
//! | Hits@k | [`hits_at_k`](crate::kge::hits_at_k) | Higher is better |
//! | Mean Rank | [`mean_rank`](crate::kge::mean_rank) | Lower is better |
//! | Adjusted Mean Rank | [`adjusted_mean_rank`](crate::kge::adjusted_mean_rank) | < 1.0 is better than random |
//!
//! # Example
//!
//! ```no_run
//! use lattix::kge::{load_dataset, mean_reciprocal_rank, hits_at_k};
//! use std::path::Path;
//!
//! let ds = load_dataset(Path::new("data/wn18rr"))?;
//! let interned = ds.into_interned();
//! println!("{} entities, {} relations", interned.num_entities(), interned.num_relations());
//! println!("{} train, {} valid, {} test",
//!     interned.train.len(), interned.valid.len(), interned.test.len());
//!
//! // After running evaluation and collecting ranks:
//! let ranks = vec![1, 3, 1, 7, 2];
//! println!("MRR: {:.3}", mean_reciprocal_rank(&ranks));
//! println!("Hits@10: {:.3}", hits_at_k(&ranks, 10));
//! # Ok::<(), lattix::Error>(())
//! ```

mod dataset;
mod filter;
mod metrics;
mod triple;

pub use dataset::{Dataset, InternedDataset, Vocab};
pub use filter::FilterIndex;
pub use metrics::{
    adjusted_mean_rank, hits_at_k, mean_rank, mean_reciprocal_rank, per_relation_mrr,
    realistic_rank, try_adjusted_mean_rank, try_hits_at_k, try_mean_rank, try_mean_reciprocal_rank,
    try_realistic_rank, MetricError,
};
pub use triple::{Triple, TripleIds};

use crate::{Error, Result};
use std::path::Path;

/// Load a KGE benchmark dataset from a directory.
///
/// Expects `train.txt`, `valid.txt`, `test.txt` in the directory. Each file
/// contains tab-separated or whitespace-separated triples (`head relation tail`),
/// one per line. Empty lines and lines starting with `#` are skipped.
///
/// Returns an error if any file is missing or contains malformed lines.
///
/// # Example
///
/// ```no_run
/// use lattix::kge::load_dataset;
/// use std::path::Path;
///
/// let ds = load_dataset(Path::new("data/wn18rr"))?;
/// println!("{} training triples", ds.train.len());
/// # Ok::<(), lattix::Error>(())
/// ```
pub fn load_dataset(path: &Path) -> Result<Dataset> {
    let train = load_triples(&path.join("train.txt"))?;
    let valid = load_triples(&path.join("valid.txt"))?;
    let test = load_triples(&path.join("test.txt"))?;
    Ok(Dataset::new(train, valid, test))
}

/// Load triples from a single file.
///
/// Supports tab-separated and whitespace-separated formats. Skips empty lines
/// and comment lines (starting with `#`). Returns an error on malformed lines.
pub fn load_triples(path: &Path) -> Result<Vec<Triple>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    if !path.exists() {
        return Err(Error::MissingFile(format!("{}", path.display())));
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut triples = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = if trimmed.contains('\t') {
            trimmed.split('\t').collect()
        } else {
            trimmed.split_whitespace().collect()
        };

        if parts.len() == 3 {
            triples.push(Triple::new(parts[0], parts[1], parts[2]));
        } else {
            return Err(Error::InvalidFormat(format!(
                "{}:{}: expected 3 fields (head, relation, tail), got {}",
                path.display(),
                line_num + 1,
                parts.len()
            )));
        }
    }

    Ok(triples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_triples_tab_separated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"e1\tr1\te2\ne3\tr2\te4\n")
            .unwrap();
        let triples = load_triples(&path).unwrap();
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].head, "e1");
        assert_eq!(triples[0].relation, "r1");
        assert_eq!(triples[0].tail, "e2");
    }

    #[test]
    fn load_triples_whitespace_separated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"e1 r1 e2\n")
            .unwrap();
        let triples = load_triples(&path).unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn load_triples_skips_comments_and_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"# comment\n\ne1\tr1\te2\n")
            .unwrap();
        let triples = load_triples(&path).unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn load_triples_errors_on_malformed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"e1 r1\n")
            .unwrap();
        assert!(load_triples(&path).is_err());
    }

    #[test]
    fn load_triples_errors_on_missing_file() {
        let path = Path::new("/nonexistent/test.txt");
        assert!(load_triples(path).is_err());
    }

    #[test]
    fn load_dataset_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        for name in &["train.txt", "valid.txt", "test.txt"] {
            std::fs::File::create(dir.path().join(name))
                .unwrap()
                .write_all(b"e1\tr1\te2\n")
                .unwrap();
        }
        let ds = load_dataset(dir.path()).unwrap();
        assert_eq!(ds.train.len(), 1);
        assert_eq!(ds.valid.len(), 1);
        assert_eq!(ds.test.len(), 1);
    }

    #[test]
    fn kge_triple_to_core_triple_conversion() {
        let kge_t = Triple::new("Alice", "knows", "Bob");
        let core_t: crate::Triple = kge_t.into();
        assert_eq!(core_t.subject().as_str(), "Alice");
        assert_eq!(core_t.predicate().as_str(), "knows");
        assert_eq!(core_t.object().as_str(), "Bob");
    }

    #[test]
    fn core_triple_to_kge_triple_conversion() {
        let core_t = crate::Triple::new("Alice", "knows", "Bob");
        let kge_t: Triple = core_t.into();
        assert_eq!(kge_t.head, "Alice");
        assert_eq!(kge_t.relation, "knows");
        assert_eq!(kge_t.tail, "Bob");
    }
}
