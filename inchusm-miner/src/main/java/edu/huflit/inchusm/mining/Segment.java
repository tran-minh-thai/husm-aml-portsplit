package edu.huflit.inchusm.mining;

/**
 * Per-snapshot evaluation slice of one pattern  -  the unit IncHUSM accumulates.
 *
 * Because snapshots are pairwise DISJOINT (Lemma L1: no occurrence and no ESV
 * ball crosses a snapshot), the union-graph quantities decompose additively:
 *
 *   |F_union(v)|  = sum over snapshots of columnCounts[v]   (images are disjoint)
 *   ESV maxima of the union = concatenation of per-snapshot maxima lists
 *
 * So a segment only needs COUNTS and the two maxima lists  -  no image sets, no
 * embeddings. Old segments are immutable forever (add-only stream), which is
 * the memory story of IncHUSM versus embedding-list approaches.
 */
public record Segment(int[] columnCounts, double[] maxAll, double[] maxGe) {

    /** True when the pattern has no MER-valid occurrence in this snapshot. */
    public boolean isEmpty() {
        return columnCounts.length == 0 || columnCounts[0] == 0;
    }
}
