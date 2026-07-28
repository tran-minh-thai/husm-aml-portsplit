package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.csp.BallCache;
import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.dfs.MinCode;
import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.mining.ValueCompute.Evaluation;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * IncHUSM  -  incremental high-utility subgraph mining over a stream of pairwise
 * disjoint, add-only snapshots (Elliptic's 49 time steps).
 *
 * Each {@link #update} folds one new snapshot into a persistent pattern store
 * without re-mining earlier snapshots. Correctness rests on three invariants,
 * all checked in tests against static re-run on the accumulated union:
 *
 *   INV1  every stored pattern's u/CEU equals static HUSM on the union (Lemma L1
 *         additivity: {@link PatternState} sums disjoint per-snapshot segments).
 *   INV2  the reported high set after snapshot t equals static HUSM on G_1..t.
 *   INV3  with an ABSOLUTE threshold, value and bound are monotone non-decreasing
 *         in t, so a pattern's state only moves candidate -> high, never back;
 *         {@code firstHighSnapshot} is its permanent alert time (append-only).
 *
 * Guided DFS each update: descend the union candidate tree (bound >= threshold)
 * but touch only the arriving snapshot  -  a pattern absent from G_t whose subtree
 * was already expanded is frozen (Lemma L3) and skipped. New patterns are
 * back-filled over their parent's occurrence snapshots only (projection),
 * never the whole history.
 */
public final class IncHUSM {

    public record Config(double minValue, int mer, HUSM.Bound bound,
                         Measure measure, int maxEdges) {
    }

    public static final class Stats {
        public int snapshots;
        public long patternsTracked;      // states currently in the store
        public long highTotal;            // patterns that have reached the threshold
        public long segmentsComputedLast; // CSP evaluations in the last update
        public long segmentsComputedTotal;
        public long lastUpdateMs;
        public long totalMs;
    }

    private final Config cfg;
    private final UtilityTable ext;
    private final RelativeThreshold rel;   // IncHUSM-mu relative threshold; null = fixed minValue
    private final int window;              // sliding-window length; 0 = unbounded (add-only)
    private double scale = 0.0;            // accumulated external-utility weight of the data seen
    private final List<Graph> snapshots = new ArrayList<>();
    private final List<BallCache> balls = new ArrayList<>();
    private final List<Double> snapshotScale = new ArrayList<>();
    private final TreeSet<Graph.EdgeType> globalTypes = new TreeSet<>();
    private final Map<String, PatternState> trie = new HashMap<>();
    private final Stats stats = new Stats();

    private int oldest = 1;       // smallest snapshot id still inside the window
    private int t = 0;
    private Graph g;              // current snapshot
    private BallCache ballsT;     // current snapshot's ball cache
    private Alphabet alpha;       // retained-by-bound alphabet for this update
    private long segThisUpdate;
    private List<PatternState> newlyHigh;

    public IncHUSM(Config cfg, UtilityTable ext) {
        this(cfg, ext, null, 0);
    }

    /** IncHUSM-mu: a relative (data-scaled) threshold with a pre-large band; null = fixed minValue. */
    public IncHUSM(Config cfg, UtilityTable ext, RelativeThreshold rel) {
        this(cfg, ext, rel, 0);
    }

    /**
     * Sliding-window IncHUSM: keep only the last {@code window} snapshots, evicting the
     * oldest after each update ({@code window <= 0} keeps the add-only behaviour).
     *
     * Under a window the high set is a function of the window, so the add-only
     * monotonicity (INV3) no longer holds: a pattern can leave the high set when the
     * snapshot carrying its value is evicted. {@link #highPatterns} therefore recomputes
     * membership from the current segments instead of trusting firstHighSnapshot.
     */
    public IncHUSM(Config cfg, UtilityTable ext, RelativeThreshold rel, int window) {
        this.cfg = cfg;
        this.ext = ext;
        this.rel = rel;
        this.window = window;
    }

    /** Effective absolute lower threshold at the current data scale (the fixed minValue if rel is null). */
    private double minVal() {
        return rel != null ? rel.lower(scale) : cfg.minValue();
    }

    /** Accumulated data scale (external-utility weight) seen so far; the base for mu * scale. */
    public double scale() {
        return scale;
    }

    /** The relative threshold, or null in fixed-threshold mode. */
    public RelativeThreshold relativeThreshold() {
        return rel;
    }

    public Stats stats() {
        return stats;
    }

    /** Live view of the pattern store (used by tests/CLI to read the high set). */
    public Map<String, PatternState> store() {
        return trie;
    }

    /** All patterns currently at or above the threshold (the reported high set). */
    public List<PatternState> highPatterns() {
        List<PatternState> out = new ArrayList<>();
        for (PatternState st : trie.values()) {
            // Add-only: once high, always high (INV3), so the alert time decides.
            // Windowed: value can fall when a snapshot is evicted, so re-test it.
            boolean high = window > 0
                    ? cfg.measure().value(st.evaluation()) >= minVal()
                    : st.firstHighSnapshot >= 0;
            if (high) out.add(st);
        }
        return out;
    }

    /**
     * Fold one snapshot into the store; returns the patterns that reached the
     * threshold FOR THE FIRST TIME at this snapshot (append-only alert stream).
     */
    public List<PatternState> update(Graph snapshot) {
        long t0 = System.nanoTime();
        t++;
        g = snapshot;
        ballsT = new BallCache(g, cfg.mer());
        snapshots.add(g);
        balls.add(ballsT);
        globalTypes.addAll(g.distinctEdgeTypes());
        double sScale = 0;
        for (int v = 0; v < g.n(); v++) sScale += ext.of(g.vlabel(v));
        snapshotScale.add(sScale);
        scale += sScale;                                                 // grow the data scale
        segThisUpdate = 0;
        newlyHigh = new ArrayList<>();

        // Phase 1  -  seeds (1-edge patterns): append this snapshot, collect the
        // bound-retained alphabet (safe prune: any candidate's first edge is a
        // 1-edge whose bound >= the candidate's bound).
        List<Graph.EdgeType> retained = new ArrayList<>();
        for (Graph.EdgeType type : globalTypes) {
            DFSCode c1 = DFSCode.single(new EdgeTuple(0, 1, type.la(), type.le(), type.lb()));
            PatternState st = trie.get(c1.key());
            if (st == null) {
                st = new PatternState(c1, ext);
                st.append(segment(g, c1, ballsT), t);   // new type => only occurs now
                if (!st.everOccurred()) continue;
                trie.put(c1.key(), st);
            } else if (!st.hasEvaluated(t)) {
                st.append(segment(g, c1, ballsT), t);
            }
            Evaluation ev = st.evaluation();
            maybeMarkHigh(st, ev);
            if (cfg.measure().bound(ev, cfg.bound()) >= minVal()) retained.add(type);
        }
        alpha = new Alphabet(retained);

        // Phase 2  -  descend from retained seeds through the candidate tree.
        for (Graph.EdgeType type : retained) {
            DFSCode c1 = DFSCode.single(new EdgeTuple(0, 1, type.la(), type.le(), type.lb()));
            expand(c1, trie.get(c1.key()));
        }

        if (window > 0) evictThrough(t - window);

        stats.snapshots = t;
        stats.patternsTracked = trie.size();
        stats.highTotal = highPatterns().size();
        stats.segmentsComputedLast = segThisUpdate;
        stats.segmentsComputedTotal += segThisUpdate;
        stats.lastUpdateMs = (System.nanoTime() - t0) / 1_000_000;
        stats.totalMs += stats.lastUpdateMs;
        return newlyHigh;
    }

    /**
     * Drop every snapshot with id <= lastToDrop out of the window (FIFO).
     *
     * By Lemma L1 the window value of a pattern is the combination of exactly the
     * segments of the snapshots still inside it, so evicting a snapshot's segment from
     * every pattern leaves each stored value equal to static mining on the window union.
     * A pattern with no remaining snapshot is dropped, which is what bounds the state;
     * the snapshot graph and its ball cache are released for the same reason.
     */
    private void evictThrough(int lastToDrop) {
        for (int s = oldest; s <= lastToDrop; s++) {
            final int snap = s;
            trie.values().removeIf(st -> {
                st.evict(snap);
                st.forgetEvaluation(snap);   // pair is gone: drop its bookkeeping too
                return !st.everOccurred();   // outside the window entirely
            });
            snapshots.set(s - 1, null);
            balls.set(s - 1, null);
            scale -= snapshotScale.get(s - 1);
            snapshotScale.set(s - 1, 0.0);
            oldest = s + 1;
        }
    }

    /** Generate children of a candidate and recurse; skip frozen subtrees (L3). */
    private void expand(DFSCode code, PatternState st) {
        if (cfg.measure().bound(st.evaluation(), cfg.bound()) < minVal()) return;
        if (cfg.maxEdges() > 0 && code.size() >= cfg.maxEdges()) return;

        boolean expandedBefore = st.lastExpandedSnapshot >= 0 && st.lastExpandedSnapshot < t;
        if (expandedBefore && !st.occursIn(t)) return; // Lemma L3: nothing new below

        BitSet parentGs = st.snapshots();
        for (EdgeTuple e : Extensions.rightmost(code, alpha)) {
            DFSCode child = code.extend(e);
            if (!MinCode.isMin(child)) continue; // anti-duplicate prune
            visitChild(child, parentGs);
        }
        st.lastExpandedSnapshot = t;
    }

    /** Ensure a child's state (filling every snapshot it still lacks), then recurse. */
    private void visitChild(DFSCode code, BitSet parentGs) {
        PatternState st = trie.get(code.key());
        boolean isNew = st == null;
        if (isNew) st = new PatternState(code, ext);

        // Fill every parent snapshot this pattern has not been evaluated on yet (child  is a subset of 
        // parent, so GS(child)  is a subset of  GS(parent) and these are the only candidates). For a new
        // pattern this is the back-fill; for an existing one it closes GAPS, which matters
        // once the bound is not monotone: under a sliding window an ancestor's CEU can dip
        // below the threshold, freezing this subtree for several snapshots, and then rise
        // again. Appending only the current snapshot would silently lose the segments of
        // that gap. Add-only never needs this because CEU only grows there, so a revived
        // branch is always brand new.
        for (int s = parentGs.nextSetBit(0); s >= 0 && s <= t; s = parentGs.nextSetBit(s + 1)) {
            if (st.hasEvaluated(s)) continue;
            st.append(segment(snapshots.get(s - 1), code, balls.get(s - 1)), s);
        }

        if (isNew) {
            if (!st.everOccurred()) return; // never occurs in the window => don't store
            trie.put(code.key(), st);
        }
        Evaluation ev = st.evaluation();
        maybeMarkHigh(st, ev);
        expand(code, st);
    }

    private void maybeMarkHigh(PatternState st, Evaluation ev) {
        if (st.firstHighSnapshot < 0 && cfg.measure().value(ev) >= minVal()) {
            st.firstHighSnapshot = t;
            newlyHigh.add(st);
        }
    }

    private Segment segment(Graph graph, DFSCode code, BallCache b) {
        segThisUpdate++;
        return ValueCompute.segment(graph, code, ext, b);
    }
}
