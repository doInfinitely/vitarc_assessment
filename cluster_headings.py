"""
Embed clinical document section headings and explore agglomerative clustering
at multiple distance thresholds.
"""

import re
from glob import glob
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

# ── Step 1: Extract ## and ### headings from each markdown file ───────────────

md_files = sorted(glob("markdown_ground_truth/*.md"))
headings: list[tuple[str, str, int]] = []  # (heading_text, source_doc, level)

for fpath in md_files:
    doc_name = Path(fpath).stem
    text = Path(fpath).read_text()
    for m in re.finditer(r"^(#{2,3}) (.+)$", text, re.MULTILINE):
        level = len(m.group(1))
        headings.append((m.group(2).strip(), doc_name, level))

print(f"Found {len(headings)} headings across {len(md_files)} documents:\n")
for h, doc, level in headings:
    prefix = "##" if level == 2 else "###"
    print(f"  [{doc}]  {prefix} {h}")

# ── Step 2: Embed with all-MiniLM-L6-v2 ──────────────────────────────────────

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [h for h, _, _ in headings]
embeddings = model.encode(texts, normalize_embeddings=True)

# ── Step 3: Pairwise cosine distance matrix ───────────────────────────────────

dist_matrix = cosine_distances(embeddings)

print("\n\n── Pairwise Cosine Distance Matrix ─────────────────────────────────\n")

# Short labels for display
labels = [f"{'##' if lv == 2 else '###'} {h} [{doc}]" for h, doc, lv in headings]
max_label = max(len(l) for l in labels)

# Header row
header = " " * (max_label + 2)
for i in range(len(labels)):
    header += f"{i:>6}"
print(header)

for i, label in enumerate(labels):
    row = f"{label:<{max_label}}  "
    for j in range(len(labels)):
        row += f"{dist_matrix[i, j]:6.3f}"
    print(row)

# Index legend
print(f"\nIndex legend:")
for i, label in enumerate(labels):
    print(f"  {i:>2}: {label}")

# ── Step 4: Determine threshold bounds ────────────────────────────────────────

# Lower bound: smallest cosine distance between headings from the SAME document
min_intra = float("inf")
min_intra_pair = ("", "")
for i in range(len(headings)):
    for j in range(i + 1, len(headings)):
        if headings[i][1] == headings[j][1] and headings[i][2] == headings[j][2]:  # same doc & level
            if dist_matrix[i, j] < min_intra:
                min_intra = dist_matrix[i, j]
                min_intra_pair = (labels[i], labels[j])

# Upper bound: distance between "Clinical Correlation" and "Clinical Indication"
# Find indices
idx_correlation = None
idx_indication = None
for i, (h, doc, lv) in enumerate(headings):
    if h == "Clinical Correlation":
        idx_correlation = i
    if h == "Clinical Indication" and idx_indication is None:
        idx_indication = i  # take first occurrence

upper_bound = dist_matrix[idx_correlation, idx_indication]

print("\n\n── Threshold Bounds ────────────────────────────────────────────────\n")
print(f"Lower bound (min intra-doc distance): {min_intra:.4f}")
print(f"  Between: {min_intra_pair[0]}")
print(f"       and {min_intra_pair[1]}")
print(f"\nUpper bound (Clinical Correlation ↔ Clinical Indication): {upper_bound:.4f}")
print(f"  Between: {labels[idx_correlation]}")
print(f"       and {labels[idx_indication]}")

# Generate ~5 evenly-spaced thresholds between bounds
n_thresholds = 5
thresholds = np.linspace(min_intra, upper_bound, n_thresholds + 2)  # include endpoints
# Keep only the interior points plus bounds
thresholds = thresholds.tolist()

print(f"\nThresholds to explore ({len(thresholds)}):")
for i, t in enumerate(thresholds):
    tag = ""
    if i == 0:
        tag = " (lower bound)"
    elif i == len(thresholds) - 1:
        tag = " (upper bound)"
    print(f"  {i + 1}. {t:.4f}{tag}")

# ── Step 5: Agglomerative clustering at each threshold ────────────────────────

print("\n\n── Clustering Results ──────────────────────────────────────────────")

for t_idx, threshold in enumerate(thresholds):
    print(f"\n{'─' * 70}")
    print(f"Threshold {t_idx + 1}: {threshold:.4f}")
    print(f"{'─' * 70}")

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    cluster_labels = clustering.fit_predict(embeddings)

    # Group headings by cluster
    clusters: dict[int, list[str]] = {}
    for i, cl in enumerate(cluster_labels):
        clusters.setdefault(cl, []).append(labels[i])

    # Sort clusters by size descending, then by cluster id
    for cl_id in sorted(clusters, key=lambda c: (-len(clusters[c]), c)):
        members = clusters[cl_id]
        if len(members) == 1:
            print(f"  Cluster {cl_id}: {members[0]}")
        else:
            print(f"  Cluster {cl_id} ({len(members)} members):")
            for m in members:
                print(f"    - {m}")

    n_clusters = len(clusters)
    n_singletons = sum(1 for v in clusters.values() if len(v) == 1)
    print(f"\n  Summary: {n_clusters} clusters ({n_singletons} singletons)")

    # Validation checks
    if t_idx == 0:
        # At lower bound: check no same-doc headings merged
        violations = []
        for cl_id, members in clusters.items():
            docs_in_cluster = [m.split("[")[1].rstrip("]") for m in members]
            if len(docs_in_cluster) != len(set(docs_in_cluster)):
                violations.append((cl_id, members))
        if violations:
            print(f"  WARNING: Same-doc headings merged at lower bound!")
            for cl_id, members in violations:
                print(f"    Cluster {cl_id}: {members}")
        else:
            print(f"  OK: No same-doc headings merged (lower bound OK)")

    if t_idx == len(thresholds) - 1:
        # At upper bound: check Clinical Correlation and Clinical Indication together
        corr_cluster = cluster_labels[idx_correlation]
        ind_cluster = cluster_labels[idx_indication]
        if corr_cluster == ind_cluster:
            print(f"  OK: Clinical Correlation & Clinical Indication in same cluster (upper bound OK)")
        else:
            print(f"  WARNING: Clinical Correlation & Clinical Indication NOT merged at upper bound!")

# ── Step 6: Merge-by-merge scan above upper bound ────────────────────────────

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

condensed = squareform(dist_matrix)
Z = linkage(condensed, method="average")

print("\n\n── Merges above upper bound (each step collapses two clusters) ───\n")

n = len(labels)
node_members: dict[int, list[int]] = {i: [i] for i in range(n)}

def fmt_group(members: list[str]) -> str:
    if len(members) == 1:
        return members[0]
    return "{" + ", ".join(members) + "}"

for step, row in enumerate(Z):
    left, right, dist, size = int(row[0]), int(row[1]), row[2], int(row[3])
    new_id = n + step
    merged_leaves = node_members[left] + node_members[right]
    node_members[new_id] = merged_leaves

    if dist <= upper_bound:
        continue

    left_members = [labels[i] for i in node_members[left]]
    right_members = [labels[i] for i in node_members[right]]

    remaining = n - 1 - step
    print(f"  Distance {dist:.4f}  ->  {remaining} clusters remaining")
    print(f"    MERGE: {fmt_group(left_members)}")
    print(f"       +   {fmt_group(right_members)}")
    print()

# ── Step 7: PCA 2D projection ────────────────────────────────────────────────

pca = PCA(n_components=2)
coords_2d = pca.fit_transform(embeddings)

print(f"\n\n── PCA 2D Coordinates ──────────────────────────────────────────────\n")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"\n{'Heading':<50} {'Doc':<28} {'PC1':>8} {'PC2':>8}")
print("─" * 96)
for i, (h, doc, lv) in enumerate(headings):
    prefix = "##" if lv == 2 else "###"
    display = f"{prefix} {h}"
    print(f"{display:<50} {doc:<28} {coords_2d[i, 0]:8.4f} {coords_2d[i, 1]:8.4f}")
