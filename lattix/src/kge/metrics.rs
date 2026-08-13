//! Rank-based evaluation metrics for link prediction.
//!
//! All functions take 1-indexed ranks (rank 1 = best possible).

use std::fmt;

/// Invalid input to a rank metric.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetricError {
    /// A rank was zero, but ranks are 1-indexed.
    ZeroRank {
        /// Position of the invalid rank.
        index: usize,
    },
    /// A score was NaN or infinite.
    NonFiniteScore {
        /// Position in the candidate score slice, or `None` for the true score.
        index: Option<usize>,
    },
    /// The true score was not present among the candidate scores.
    MissingTrueCandidate,
    /// The constant candidate count was zero.
    ZeroCandidateCount,
    /// A rank exceeded the constant candidate count.
    RankExceedsCandidateCount {
        /// Position of the invalid rank.
        index: usize,
        /// Invalid rank.
        rank: usize,
        /// Number of candidates used for every query.
        candidate_count: usize,
    },
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRank { index } => write!(f, "rank at index {index} is zero"),
            Self::NonFiniteScore { index: Some(index) } => {
                write!(f, "candidate score at index {index} is not finite")
            }
            Self::NonFiniteScore { index: None } => write!(f, "true score is not finite"),
            Self::MissingTrueCandidate => {
                write!(f, "true score is absent from the candidate scores")
            }
            Self::ZeroCandidateCount => write!(f, "candidate count is zero"),
            Self::RankExceedsCandidateCount {
                index,
                rank,
                candidate_count,
            } => write!(
                f,
                "rank {rank} at index {index} exceeds candidate count {candidate_count}"
            ),
        }
    }
}

impl std::error::Error for MetricError {}

fn validate_ranks(ranks: &[usize]) -> Result<(), MetricError> {
    match ranks.iter().position(|&rank| rank == 0) {
        Some(index) => Err(MetricError::ZeroRank { index }),
        None => Ok(()),
    }
}

/// Mean Reciprocal Rank: average of `1/rank` over all queries.
///
/// Higher is better. Range: (0, 1]. Returns 0.0 for empty input.
///
/// # Panics
///
/// Panics if a rank is zero. Use [`try_mean_reciprocal_rank`] to handle invalid input.
pub fn mean_reciprocal_rank(ranks: &[usize]) -> f64 {
    try_mean_reciprocal_rank(ranks).expect("mean_reciprocal_rank requires 1-indexed ranks")
}

/// Checked form of [`mean_reciprocal_rank`].
pub fn try_mean_reciprocal_rank(ranks: &[usize]) -> Result<f64, MetricError> {
    validate_ranks(ranks)?;
    if ranks.is_empty() {
        return Ok(0.0);
    }
    Ok(ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / ranks.len() as f64)
}

/// Hits@k: fraction of queries where the correct answer ranks at or above `k`.
///
/// Higher is better. Range: [0, 1]. Returns 0.0 for empty input.
///
/// # Panics
///
/// Panics if a rank is zero. Use [`try_hits_at_k`] to handle invalid input.
pub fn hits_at_k(ranks: &[usize], k: usize) -> f64 {
    try_hits_at_k(ranks, k).expect("hits_at_k requires 1-indexed ranks")
}

/// Checked form of [`hits_at_k`].
pub fn try_hits_at_k(ranks: &[usize], k: usize) -> Result<f64, MetricError> {
    validate_ranks(ranks)?;
    if ranks.is_empty() {
        return Ok(0.0);
    }
    Ok(ranks.iter().filter(|&&r| r <= k).count() as f64 / ranks.len() as f64)
}

/// Mean Rank: arithmetic mean of all ranks.
///
/// Lower is better. Returns 0.0 for empty input.
///
/// # Panics
///
/// Panics if a rank is zero. Use [`try_mean_rank`] to handle invalid input.
pub fn mean_rank(ranks: &[usize]) -> f64 {
    try_mean_rank(ranks).expect("mean_rank requires 1-indexed ranks")
}

/// Checked form of [`mean_rank`].
pub fn try_mean_rank(ranks: &[usize]) -> Result<f64, MetricError> {
    validate_ranks(ranks)?;
    if ranks.is_empty() {
        return Ok(0.0);
    }
    // Convert before summing so valid ranks cannot overflow a `usize` accumulator.
    Ok(ranks.iter().map(|&rank| rank as f64).sum::<f64>() / ranks.len() as f64)
}

/// Compute the realistic rank of the true entity given all scores.
///
/// Realistic rank (PyKEEN convention) is the mean of optimistic and pessimistic:
/// - Optimistic: number of entities with strictly better score + 1
/// - Pessimistic: number of entities with score at least as good
///
/// `true_score` is the score of the correct entity. `all_scores` includes
/// all candidate scores (including the true entity, excluding filtered entities).
/// Lower scores are assumed better (distance convention).
///
/// # Panics
///
/// Panics if a score is not finite or `true_score` is absent from `all_scores`.
/// Use [`try_realistic_rank`] to handle invalid input.
pub fn realistic_rank(all_scores: &[f32], true_score: f32) -> f64 {
    try_realistic_rank(all_scores, true_score)
        .expect("realistic_rank requires finite scores containing the true score")
}

/// Checked form of [`realistic_rank`].
pub fn try_realistic_rank(all_scores: &[f32], true_score: f32) -> Result<f64, MetricError> {
    if !true_score.is_finite() {
        return Err(MetricError::NonFiniteScore { index: None });
    }
    if let Some(index) = all_scores.iter().position(|score| !score.is_finite()) {
        return Err(MetricError::NonFiniteScore { index: Some(index) });
    }
    if !all_scores.contains(&true_score) {
        return Err(MetricError::MissingTrueCandidate);
    }

    let mut strictly_better = 0usize;
    let mut at_least_as_good = 0usize;
    for &s in all_scores {
        if s < true_score {
            strictly_better += 1;
        }
        if s <= true_score {
            at_least_as_good += 1;
        }
    }
    let optimistic = strictly_better + 1;
    let pessimistic = at_least_as_good;
    Ok((optimistic as f64 + pessimistic as f64) / 2.0)
}

/// Adjusted Mean Rank: `mean_rank / expected_random_mean_rank`.
///
/// The expected mean rank under a uniform random model is `(candidate_count + 1) / 2`.
/// AMR < 1.0 means better than random; AMR = 1.0 means random performance.
///
/// Returns 0.0 for empty input.
///
/// # Panics
///
/// Panics if `candidate_count` is zero or a rank is outside
/// `1..=candidate_count`. Use [`try_adjusted_mean_rank`] to handle invalid input.
pub fn adjusted_mean_rank(ranks: &[usize], candidate_count: usize) -> f64 {
    try_adjusted_mean_rank(ranks, candidate_count)
        .expect("adjusted_mean_rank requires valid ranks and a nonzero candidate count")
}

/// Checked form of [`adjusted_mean_rank`].
///
/// `candidate_count` is the number of candidates used for every query. If the
/// candidate count varies by query, compute each query's adjustment separately.
pub fn try_adjusted_mean_rank(ranks: &[usize], candidate_count: usize) -> Result<f64, MetricError> {
    validate_ranks(ranks)?;
    if candidate_count == 0 {
        return Err(MetricError::ZeroCandidateCount);
    }
    if let Some((index, &rank)) = ranks
        .iter()
        .enumerate()
        .find(|(_, rank)| **rank > candidate_count)
    {
        return Err(MetricError::RankExceedsCandidateCount {
            index,
            rank,
            candidate_count,
        });
    }
    if ranks.is_empty() {
        return Ok(0.0);
    }
    let mr = try_mean_rank(ranks)?;
    let expected = (candidate_count as f64 + 1.0) / 2.0;
    Ok(mr / expected)
}

/// Compute per-relation MRR from `(relation_id, rank)` pairs.
///
/// Returns a map from relation ID to MRR. Useful for diagnosing which
/// relation types a model handles well vs poorly.
pub fn per_relation_mrr(rel_ranks: &[(usize, usize)]) -> std::collections::HashMap<usize, f64> {
    let mut grouped: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for &(rel, rank) in rel_ranks {
        grouped.entry(rel).or_default().push(rank);
    }
    grouped
        .into_iter()
        .map(|(rel, ranks)| (rel, mean_reciprocal_rank(&ranks)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrr_basic() {
        let ranks = vec![1, 2, 4];
        let mrr = mean_reciprocal_rank(&ranks);
        // (1/1 + 1/2 + 1/4) / 3 = 1.75 / 3
        assert!((mrr - 0.5833).abs() < 0.001);
    }

    #[test]
    fn hits_at_k_basic() {
        let ranks = vec![1, 2, 5, 10, 20];
        assert!((hits_at_k(&ranks, 10) - 0.8).abs() < 1e-9);
        assert!((hits_at_k(&ranks, 1) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn mean_rank_basic() {
        let ranks = vec![1, 3, 5];
        assert!((mean_rank(&ranks) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn adjusted_mean_rank_basic() {
        let ranks = vec![1, 1, 1]; // MR = 1.0, expected = 50.5
        let amr = adjusted_mean_rank(&ranks, 100);
        assert!((amr - 1.0 / 50.5).abs() < 1e-9);
    }

    #[test]
    fn realistic_rank_no_ties() {
        // Scores: [0.1, 0.5, 0.3, 0.9], true = 0.3
        // strictly_better = 1 (0.1), at_least_as_good = 2 (0.1, 0.3)
        // optimistic = 2, pessimistic = 2, realistic = 2.0
        let scores = vec![0.1, 0.5, 0.3, 0.9];
        assert!((realistic_rank(&scores, 0.3) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn realistic_rank_with_ties() {
        // Scores: [0.3, 0.3, 0.3, 0.9], true = 0.3
        // strictly_better = 0, at_least_as_good = 3
        // optimistic = 1, pessimistic = 3, realistic = 2.0
        let scores = vec![0.3, 0.3, 0.3, 0.9];
        assert!((realistic_rank(&scores, 0.3) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn realistic_rank_best() {
        // True entity is the best scorer
        let scores = vec![0.1, 0.5, 0.9];
        assert!((realistic_rank(&scores, 0.1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn realistic_rank_worst() {
        // True entity is the worst scorer
        let scores = vec![0.1, 0.5, 0.9];
        // strictly_better = 2, at_least_as_good = 3
        // optimistic = 3, pessimistic = 3, realistic = 3.0
        assert!((realistic_rank(&scores, 0.9) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn empty_ranks() {
        assert_eq!(mean_reciprocal_rank(&[]), 0.0);
        assert_eq!(hits_at_k(&[], 10), 0.0);
        assert_eq!(mean_rank(&[]), 0.0);
        assert_eq!(adjusted_mean_rank(&[], 100), 0.0);
    }

    #[test]
    fn checked_metrics_reject_zero_rank() {
        let expected = Err(MetricError::ZeroRank { index: 1 });
        assert_eq!(try_mean_reciprocal_rank(&[1, 0, 2]), expected);
        assert_eq!(try_hits_at_k(&[1, 0, 2], 2), expected);
        assert_eq!(try_mean_rank(&[1, 0, 2]), expected);
        assert_eq!(try_adjusted_mean_rank(&[1, 0, 2], 3), expected);
    }

    #[test]
    fn mean_rank_does_not_overflow_usize() {
        let ranks = [usize::MAX, usize::MAX];
        assert!(mean_rank(&ranks).is_finite());
        assert_eq!(mean_rank(&ranks), usize::MAX as f64);
    }

    #[test]
    fn checked_realistic_rank_rejects_invalid_candidate_sets() {
        assert_eq!(
            try_realistic_rank(&[0.1, f32::NAN], 0.1),
            Err(MetricError::NonFiniteScore { index: Some(1) })
        );
        assert_eq!(
            try_realistic_rank(&[0.1, 0.2], f32::INFINITY),
            Err(MetricError::NonFiniteScore { index: None })
        );
        assert_eq!(
            try_realistic_rank(&[0.1, 0.2], 0.3),
            Err(MetricError::MissingTrueCandidate)
        );
    }

    #[test]
    fn adjusted_mean_rank_uses_one_constant_candidate_count() {
        // A random-order oracle over all ranks for four candidates has MR 2.5,
        // so its adjusted mean rank is exactly one.
        assert_eq!(try_adjusted_mean_rank(&[1, 2, 3, 4], 4), Ok(1.0));
        assert_eq!(
            try_adjusted_mean_rank(&[1, 5], 4),
            Err(MetricError::RankExceedsCandidateCount {
                index: 1,
                rank: 5,
                candidate_count: 4,
            })
        );
        assert_eq!(
            try_adjusted_mean_rank(&[1], 0),
            Err(MetricError::ZeroCandidateCount)
        );
    }
}
