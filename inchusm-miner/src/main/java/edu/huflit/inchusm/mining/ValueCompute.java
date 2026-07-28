package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.csp.BallCache;
import edu.huflit.inchusm.csp.BottleneckSolver;
import edu.huflit.inchusm.csp.CMNITable;
import edu.huflit.inchusm.csp.CSPSolver;
import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.graph.Graph;

import java.util.Arrays;
import java.util.Set;

/**
 * HUSM Algorithm 3 (ValueCompute): utility of a pattern plus the three upper
 * bounds of Defs. 12-14. All three are returned so bound choice is an ablation
 * flag, not a code path:
 *
 *   u(P)   = CMNI(P) x mean over pattern vertices of external label utility (Def. 8)
 *   IVU(P) = sum over ESVs of the max label utility inside each ball (Def. 12)
 *   EU(P)  = sum of the top-CMNI(P) of those maxima (Def. 13)
 *   CEU(P) = same but only vertices with label >= label(I0) count (Def. 14)
 *
 * Invariant (Lemma 3 + defs): u <= CEU <= EU <= IVU  -  asserted in tests.
 *
 * Split into {@link #segment} (one snapshot) and {@link #combine} (union of
 * disjoint snapshots): because snapshots share no vertex and no MER ball crosses
 * a snapshot, the union CMNI columns are the disjoint sum of per-snapshot column
 * counts and the union ESV maxima are the concatenation of per-snapshot maxima.
 * So {@code combine(segment(G1), ..., segment(Gt))} equals {@code evaluate(union)}
 * exactly, the algebraic core of IncHUSM's correctness (Lemma L1).
 */
public final class ValueCompute {

    /**
     * All measured quantities of one pattern evaluation.
     *
     * {@code bottleneck} and {@code flow} are populated only by {@link #evaluateFlow}
     * (the FLOW measure); the utility/incremental paths leave them 0, so nothing else
     * is affected. {@code flow = cmni * bottleneck} = u_flow(P).
     */
    public record Evaluation(int cmni, double utility, double ivu, double eu, double ceu,
                             double bottleneck, double flow) {

        public static final Evaluation ZERO = new Evaluation(0, 0, 0, 0, 0, 0, 0);

        public Evaluation(int cmni, double utility, double ivu, double eu, double ceu) {
            this(cmni, utility, ivu, eu, ceu, 0, 0);
        }

        public double bound(HUSM.Bound b) {
            return switch (b) {
                case IVU -> ivu;
                case EU -> eu;
                case CEU -> ceu;
            };
        }
    }

    private ValueCompute() {
    }

    /** Mean external utility of the pattern's vertex labels (Def. 8 factor). */
    public static double meanLabelUtility(DFSCode code, UtilityTable ext) {
        int nP = code.nVertices();
        double mean = 0;
        for (int v = 0; v < nP; v++) mean += ext.of(code.vertexLabel(v));
        return mean / nP;
    }

    /**
     * Per-snapshot slice of a pattern's evaluation: for each pattern vertex the
     * count of distinct images in this snapshot, and for each distinct f(I0) the
     * two ESV maxima (all labels; labels &gt;= label(I0)). No image sets or
     * embeddings are kept.
     */
    public static Segment segment(Graph g, DFSCode code, UtilityTable ext, BallCache balls) {
        CMNITable table = new CSPSolver(g, code, balls).solve();
        int nP = code.nVertices();
        int[] counts = new int[nP];
        for (int v = 0; v < nP; v++) counts[v] = table.images(v).size();
        if (counts[0] == 0) {
            return new Segment(new int[0], new double[0], new double[0]);
        }

        Set<Integer> initials = table.images(0);
        int labelI0 = code.vertexLabel(0);
        double[] maxAll = new double[initials.size()];
        double[] maxGe = new double[initials.size()];
        int idx = 0;
        for (int u0 : initials) {
            double mAll = Double.NEGATIVE_INFINITY;
            double mGe = Double.NEGATIVE_INFINITY;
            for (int x : balls.ball(u0)) {
                double ux = ext.of(g.vlabel(x));
                if (ux > mAll) mAll = ux;
                if (g.vlabel(x) >= labelI0 && ux > mGe) mGe = ux;
            }
            maxAll[idx] = mAll;
            maxGe[idx] = mGe; // never -inf: u0 itself carries label(I0)
            idx++;
        }
        return new Segment(counts, maxAll, maxGe);
    }

    /**
     * Union evaluation from disjoint-snapshot segments (Lemma L1 additivity).
     * @param unionCounts per pattern vertex, summed distinct-image counts
     * @param maxAllConcat concatenated per-snapshot all-label ESV maxima
     * @param maxGeConcat concatenated per-snapshot label&gt;=I0 ESV maxima
     */
    public static Evaluation combine(long[] unionCounts, double[] maxAllConcat,
                                     double[] maxGeConcat, double meanLabel) {
        return combine(unionCounts, maxAllConcat, maxAllConcat.length,
                maxGeConcat, maxGeConcat.length, meanLabel);
    }

    /** Length-aware combine, for IncHUSM's growable per-snapshot maxima buffers. */
    public static Evaluation combine(long[] unionCounts, double[] maxAll, int maxAllLen,
                                     double[] maxGe, int maxGeLen, double meanLabel) {
        long cmniL = Long.MAX_VALUE;
        for (long c : unionCounts) cmniL = Math.min(cmniL, c);
        int cmni = (int) Math.min(cmniL, Integer.MAX_VALUE);
        if (cmni == 0) return Evaluation.ZERO;
        double utility = cmni * meanLabel;
        double ivu = 0;
        for (int i = 0; i < maxAllLen; i++) ivu += maxAll[i];
        double eu = topKSum(maxAll, maxAllLen, cmni);
        double ceu = topKSum(maxGe, maxGeLen, cmni);
        return new Evaluation(cmni, utility, ivu, eu, ceu);
    }

    /** Convenience for the static miner: evaluate a pattern on one graph. */
    public static Evaluation evaluate(Graph g, DFSCode code, UtilityTable ext, BallCache balls) {
        Segment s = segment(g, code, ext, balls);
        if (s.isEmpty()) return Evaluation.ZERO;
        long[] counts = new long[s.columnCounts().length];
        for (int v = 0; v < counts.length; v++) counts[v] = s.columnCounts()[v];
        return combine(counts, s.maxAll(), s.maxGe(), meanLabelUtility(code, ext));
    }

    /**
     * FLOW measure: CMNI (support) times B(P) (bottleneck flow), the anti-monotone
     * money measure of the bottleneck measure. Needs no utility table and no upper bound: flow is
     * its own pruning bound. utility/ivu/eu/ceu are left 0 here.
     */
    public static Evaluation evaluateFlow(Graph g, DFSCode code, BallCache balls) {
        int cmni = new CSPSolver(g, code, balls).solve().cmni();
        if (cmni == 0) return Evaluation.ZERO;
        double bottleneck = new BottleneckSolver(g, code, balls).solve();
        return new Evaluation(cmni, 0, 0, 0, 0, bottleneck, cmni * bottleneck);
    }

    private static double topKSum(double[] values, int len, int k) {
        if (k >= len) {
            double sum = 0;
            for (int i = 0; i < len; i++) sum += values[i];
            return sum;
        }
        double[] copy = Arrays.copyOf(values, len);
        Arrays.sort(copy);
        double sum = 0;
        for (int i = len - 1; i >= len - k; i--) sum += copy[i];
        return sum;
    }
}
