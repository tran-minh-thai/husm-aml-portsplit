package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.mining.ValueCompute.Evaluation;

import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

/**
 * Accumulated cross-snapshot state of one pattern in the IncHUSM store.
 *
 * Holds only additive summaries (Lemma L1): per-vertex union image counts and the
 * concatenated per-snapshot ESV maxima  -  never embeddings. {@link #evaluation()}
 * reconstructs the exact union u/IVU/EU/CEU on demand, so it always equals what
 * static HUSM would compute on the union graph (invariant INV1).
 *
 * {@code gs} is the snapshot id set (support set / DyFSM's GS). On an add-only stream
 * appends are the only mutation. Under a sliding window {@link #evict} additionally
 * removes the oldest snapshot's contribution, which Lemma L1 makes exact: the counts
 * subtract and the ESV maxima multisets shrink by that snapshot's block.
 */
public final class PatternState {

    private final DFSCode code;
    private final int nP;
    private final double meanLabel;
    private final long[] unionCounts;
    private double[] maxAll = new double[4];
    private int maxAllLen = 0;
    private double[] maxGe = new double[4];
    private int maxGeLen = 0;
    private final BitSet gs = new BitSet();
    // Snapshots whose segment has been COMPUTED for this pattern, whether or not it occurs
    // there. Distinct from gs (which records only occurrences): it is what keeps each
    // (pattern, snapshot) pair evaluated at most once while also revealing gaps that must
    // be filled when a subtree is revisited after its bound dipped (sliding window).
    private final BitSet evaluated = new BitSet();

    // Per-snapshot block bookkeeping, needed ONLY to support eviction (sliding window).
    // Segments are appended in ascending snapshot order, so the oldest snapshot's maxima
    // always occupy the head of the two buffers; FIFO eviction is therefore a head drop.
    private final Map<Integer, int[]> countsBySnapshot = new HashMap<>();
    private final Map<Integer, int[]> blockLenBySnapshot = new HashMap<>();  // {lenAll, lenGe}

    int firstHighSnapshot = -1;   // when value first reached the threshold (alert time)
    int lastExpandedSnapshot = -1; // last update at which children were generated

    PatternState(DFSCode code, UtilityTable ext) {
        this.code = code;
        this.nP = code.nVertices();
        this.unionCounts = new long[nP];
        this.meanLabel = ValueCompute.meanLabelUtility(code, ext);
    }

    public DFSCode code() {
        return code;
    }

    /** Snapshot at which the pattern first reached the threshold, or -1. */
    public int firstHighSnapshot() {
        return firstHighSnapshot;
    }

    /** Number of snapshots containing the pattern (support = |GS|). */
    public int supportSnapshotCount() {
        return gs.cardinality();
    }

    boolean occursIn(int snapshot) {
        return gs.get(snapshot);
    }

    /** True when this (pattern, snapshot) pair has already been evaluated. */
    boolean hasEvaluated(int snapshot) {
        return evaluated.get(snapshot);
    }

    boolean everOccurred() {
        return !gs.isEmpty();
    }

    /** Append one snapshot's segment (no-op if the pattern is absent there). */
    void append(Segment seg, int snapshot) {
        if (evaluated.get(snapshot)) return;
        evaluated.set(snapshot);
        if (seg.isEmpty()) return;
        int[] c = seg.columnCounts();
        for (int v = 0; v < nP; v++) unionCounts[v] += c[v];
        maxAll = appendInto(maxAll, maxAllLen, seg.maxAll());
        maxAllLen += seg.maxAll().length;
        maxGe = appendInto(maxGe, maxGeLen, seg.maxGe());
        maxGeLen += seg.maxGe().length;
        gs.set(snapshot);
        countsBySnapshot.put(snapshot, c.clone());
        blockLenBySnapshot.put(snapshot, new int[] {seg.maxAll().length, seg.maxGe().length});
    }

    /**
     * Remove one snapshot's contribution (sliding window). Inverse of {@link #append}:
     * the counts subtract by Lemma L1 additivity, and the snapshot's ESV maxima are
     * dropped from the two multisets. Returns false if the pattern never occurred there.
     *
     * Only FIFO (oldest-first) eviction is supported: blocks are stored in ascending
     * snapshot order, so the evicted snapshot's block is the head of each buffer.
     */
    boolean evict(int snapshot) {
        if (!gs.get(snapshot)) return false;
        int oldest = gs.nextSetBit(0);
        if (snapshot != oldest) {
            throw new IllegalStateException(
                    "evict(" + snapshot + ") but oldest tracked snapshot is " + oldest
                            + "; only FIFO eviction is supported");
        }
        int[] c = countsBySnapshot.remove(snapshot);
        for (int v = 0; v < nP; v++) unionCounts[v] -= c[v];
        int[] len = blockLenBySnapshot.remove(snapshot);
        System.arraycopy(maxAll, len[0], maxAll, 0, maxAllLen - len[0]);
        maxAllLen -= len[0];
        System.arraycopy(maxGe, len[1], maxGe, 0, maxGeLen - len[1]);
        maxGeLen -= len[1];
        gs.clear(snapshot);
        evaluated.clear(snapshot);
        return true;
    }

    /** Forget an evaluation of a snapshot the pattern does not occur in (sliding window). */
    void forgetEvaluation(int snapshot) {
        if (!gs.get(snapshot)) evaluated.clear(snapshot);
    }

    public Evaluation evaluation() {
        return ValueCompute.combine(unionCounts, maxAll, maxAllLen, maxGe, maxGeLen, meanLabel);
    }

    BitSet snapshots() {
        return gs;
    }

    private static double[] appendInto(double[] buf, int len, double[] add) {
        if (len + add.length > buf.length) {
            buf = Arrays.copyOf(buf, Math.max(buf.length * 2, len + add.length));
        }
        System.arraycopy(add, 0, buf, len, add.length);
        return buf;
    }
}
