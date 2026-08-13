#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lattix::{KnowledgeGraph, Triple};

// ---------------------------------------------------------------------------
// Graph generators
// ---------------------------------------------------------------------------

/// Ring graph: node_i -> node_{i+1}, last -> first.
fn make_ring(n: usize) -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::with_capacity(n, n);
    for i in 0..n {
        let s = format!("node_{i}");
        let o = format!("node_{}", (i + 1) % n);
        kg.add_triple(Triple::new(s.as_str(), "next", o.as_str()));
    }
    kg
}

/// Random graph with deterministic seed. Each node connects to ~avg_degree random others.
fn make_random(n: usize, avg_degree: usize, seed: u64) -> KnowledgeGraph {
    let total_edges = n * avg_degree;
    let mut kg = KnowledgeGraph::with_capacity(n, total_edges);

    // Simple deterministic LCG (no dep on rand for bench helpers)
    let mut state = seed;
    let next = |s: &mut u64| -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    };

    for i in 0..n {
        let s = format!("node_{i}");
        for _ in 0..avg_degree {
            let target = (next(&mut state) as usize) % n;
            let o = format!("node_{target}");
            kg.add_triple(Triple::new(s.as_str(), "rel", o.as_str()));
        }
    }
    kg
}

/// Star graph: hub connects to all leaf nodes, leaves connect back to hub.
fn make_star(n: usize) -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::with_capacity(n, 2 * (n - 1));
    for i in 1..n {
        let leaf = format!("node_{i}");
        kg.add_triple(Triple::new("hub", "to", leaf.as_str()));
        kg.add_triple(Triple::new(leaf.as_str(), "to", "hub"));
    }
    kg
}

/// Dense fan-in/fan-out graph with several predicates per unique edge.
fn make_parallel_hubs(n: usize, parallel_edges: usize) -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::with_capacity(n, 2 * (n - 1) * parallel_edges);
    for i in 1..n {
        let leaf = format!("node_{i}");
        for edge in 0..parallel_edges {
            let predicate = format!("rel_{edge}");
            kg.add_triple(Triple::new("hub", predicate.as_str(), leaf.as_str()));
            kg.add_triple(Triple::new(leaf.as_str(), predicate.as_str(), "hub"));
        }
    }
    kg
}

/// Generate N-Triples formatted string for a ring graph of size n.
#[cfg(feature = "formats")]
fn make_ntriples_string(n: usize) -> String {
    let mut buf = String::new();
    for i in 0..n {
        let j = (i + 1) % n;
        buf.push_str(&format!(
            "<http://example.org/node_{i}> <http://example.org/next> <http://example.org/node_{j}> .\n"
        ));
    }
    buf
}

/// Generate CSV string for a ring graph of size n.
#[cfg(feature = "formats")]
fn make_csv_string(n: usize) -> String {
    let mut buf = String::new();
    for i in 0..n {
        let j = (i + 1) % n;
        buf.push_str(&format!("node_{i},next,node_{j}\n"));
    }
    buf
}

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

const SIZES: [usize; 3] = [100, 1_000, 10_000];

// ---------------------------------------------------------------------------
// Group 1: Construction
// ---------------------------------------------------------------------------

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.sample_size(50);

    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("ring", n), &n, |b, &n| {
            b.iter(|| black_box(make_ring(n)));
        });

        group.bench_with_input(BenchmarkId::new("random_d6", n), &n, |b, &n| {
            b.iter(|| black_box(make_random(n, 6, 42)));
        });

        group.bench_with_input(BenchmarkId::new("star", n), &n, |b, &n| {
            b.iter(|| black_box(make_star(n)));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: Insertion
// ---------------------------------------------------------------------------

fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.sample_size(50);

    // Single triple insertion
    group.bench_function("insert_triple", |b| {
        b.iter_batched(
            KnowledgeGraph::new,
            |mut kg| {
                kg.add_triple(Triple::new("Alice", "knows", "Bob"));
                black_box(kg)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Batch insertion
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("insert_batch", n), &n, |b, &n| {
            let triples: Vec<Triple> = (0..n)
                .map(|i| Triple::new(format!("s_{i}").as_str(), "rel", format!("o_{i}").as_str()))
                .collect();

            b.iter_batched(
                || (KnowledgeGraph::with_capacity(n * 2, n), triples.clone()),
                |(mut kg, triples)| {
                    for t in triples {
                        kg.add_triple(t);
                    }
                    black_box(kg)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: Query
// ---------------------------------------------------------------------------

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.sample_size(50);

    let kg = make_random(1_000, 6, 42);
    let ring = make_ring(1_000);

    group.bench_function("relations_from", |b| {
        b.iter(|| black_box(kg.relations_from("node_0")));
    });

    group.bench_function("relations_to", |b| {
        b.iter(|| black_box(kg.relations_to("node_0")));
    });

    group.bench_function("find_path", |b| {
        b.iter(|| black_box(ring.find_path("node_0", "node_500")));
    });

    group.bench_function("has_edge", |b| {
        b.iter(|| black_box(kg.has_edge("node_0", "node_1")));
    });

    group.bench_function("neighbors", |b| {
        b.iter(|| black_box(kg.neighbors("node_0")));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: Centrality (algo feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "algo")]
fn bench_centrality(c: &mut Criterion) {
    use lattix::algo::centrality::{
        betweenness_centrality, closeness_centrality, degree_centrality, eigenvector_centrality,
        hits, katz_centrality, BetweennessConfig, ClosenessConfig, EigenvectorConfig, HitsConfig,
        KatzConfig,
    };
    use lattix::algo::pagerank::{pagerank, PageRankConfig};

    let mut group = c.benchmark_group("centrality");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.sample_size(50);

    // Pre-build graphs outside the benchmark loop
    let graphs: Vec<(usize, KnowledgeGraph)> =
        SIZES.iter().map(|&n| (n, make_random(n, 6, 42))).collect();

    // Degree centrality: all sizes
    for (n, kg) in &graphs {
        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::new("degree", n), kg, |b, kg| {
            b.iter(|| black_box(degree_centrality(kg)));
        });
    }

    // Betweenness: 100 and 1000 only (O(VE), too slow for 10k)
    for (n, kg) in graphs.iter().filter(|(n, _)| *n <= 1_000) {
        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::new("betweenness", n), kg, |b, kg| {
            b.iter(|| black_box(betweenness_centrality(kg, BetweennessConfig::default())));
        });
    }

    // Closeness: 100 and 1000 only
    for (n, kg) in graphs.iter().filter(|(n, _)| *n <= 1_000) {
        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::new("closeness", n), kg, |b, kg| {
            b.iter(|| black_box(closeness_centrality(kg, ClosenessConfig::default())));
        });
    }

    // PageRank: all sizes
    for (n, kg) in &graphs {
        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::new("pagerank", n), kg, |b, kg| {
            b.iter(|| black_box(pagerank(kg, PageRankConfig::default())));
        });
    }

    // HITS: 100 and 1000 only
    for (n, kg) in graphs.iter().filter(|(n, _)| *n <= 1_000) {
        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::new("hits", n), kg, |b, kg| {
            b.iter(|| black_box(hits(kg, HitsConfig::default())));
        });
    }

    // Iterative algorithms on a graph where deduplicating adjacency each round is costly.
    let parallel_hubs = make_parallel_hubs(256, 16);
    group.throughput(Throughput::Elements(parallel_hubs.triple_count() as u64));
    group.bench_function("hits_parallel_hubs", |b| {
        b.iter(|| black_box(hits(&parallel_hubs, HitsConfig::default())));
    });
    group.bench_function("eigenvector_parallel_hubs", |b| {
        b.iter(|| {
            black_box(eigenvector_centrality(
                &parallel_hubs,
                EigenvectorConfig::default(),
            ))
        });
    });
    group.bench_function("katz_parallel_hubs", |b| {
        b.iter(|| black_box(katz_centrality(&parallel_hubs, KatzConfig::default())));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 5: Format roundtrip (formats feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "formats")]
fn bench_format_roundtrip(c: &mut Criterion) {
    use lattix::formats::{Csv, NTriples};

    let mut group = c.benchmark_group("format_roundtrip");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.sample_size(50);

    for &n in &[100usize, 1_000] {
        // N-Triples parse
        let nt_string = make_ntriples_string(n);
        let nt_bytes = nt_string.len() as u64;
        group.throughput(Throughput::Bytes(nt_bytes));
        group.bench_with_input(BenchmarkId::new("ntriples_parse", n), &nt_string, |b, s| {
            b.iter(|| black_box(NTriples::parse(s).unwrap()));
        });

        // N-Triples serialize
        let kg_for_nt = NTriples::parse(&nt_string).unwrap();
        group.bench_with_input(
            BenchmarkId::new("ntriples_serialize", n),
            &kg_for_nt,
            |b, kg| {
                b.iter(|| black_box(NTriples::to_string(kg).unwrap()));
            },
        );

        // CSV parse
        let csv_string = make_csv_string(n);
        let csv_bytes = csv_string.len() as u64;
        group.throughput(Throughput::Bytes(csv_bytes));
        group.bench_with_input(BenchmarkId::new("csv_parse", n), &csv_string, |b, s| {
            b.iter(|| black_box(Csv::read(s.as_bytes()).unwrap()));
        });

        // JSON serde roundtrip
        let kg_for_json = make_ring(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("json_serde_roundtrip", n),
            &kg_for_json,
            |b, kg| {
                b.iter(|| {
                    let json = serde_json::to_string(kg).unwrap();
                    let loaded: KnowledgeGraph = serde_json::from_str(&json).unwrap();
                    black_box(loaded)
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------

// Conditionally define groups based on features
#[cfg(all(feature = "algo", feature = "formats"))]
criterion_group!(
    benches,
    bench_construction,
    bench_insertion,
    bench_query,
    bench_centrality,
    bench_format_roundtrip,
);

#[cfg(all(feature = "algo", not(feature = "formats")))]
criterion_group!(
    benches,
    bench_construction,
    bench_insertion,
    bench_query,
    bench_centrality,
);

#[cfg(all(not(feature = "algo"), feature = "formats"))]
criterion_group!(
    benches,
    bench_construction,
    bench_insertion,
    bench_query,
    bench_format_roundtrip,
);

#[cfg(not(any(feature = "algo", feature = "formats")))]
criterion_group!(benches, bench_construction, bench_insertion, bench_query,);

criterion_main!(benches);
