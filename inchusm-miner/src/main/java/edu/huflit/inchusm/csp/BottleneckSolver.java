package edu.huflit.inchusm.csp;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.graph.Graph;

import java.util.ArrayList;
import java.util.List;

/**
 * Bottleneck flow of a pattern (the FLOW measure of this work, the bottleneck measure):
 *
 *   B(P) = max over MER-valid occurrences phi of  min over edges e of P  w(phi(e))
 *
 * the weakest edge of the best-funded occurrence. Together with support this defines
 * u_flow(P) = CMNI(P) x B(P), which is ANTI-MONOTONE: every occurrence of an extension
 * P' restricts to an occurrence of P, and E(P) is a subset of E(P'), so a min over the
 * larger edge set cannot exceed the min over the smaller, giving B(P') <= B(P); CMNI is
 * anti-monotone too, so the product is. Unlike HUSM's utility, u_flow needs no separate
 * upper bound: the value is its own pruning bound.
 *
 * B(P) is the MAXIMUM over occurrences, so it cannot be read off the MNI cross-marking
 * (which keeps one occurrence per image, not the best-funded one). This is a dedicated
 * branch-and-bound occurrence search sharing the CSP structure (parent/anchor/earlier
 * edges/MER ball): the running minimum edge weight only falls as a mapping grows, so a
 * partial mapping whose running minimum is already <= the best complete bottleneck found
 * cannot improve it and is pruned. Neighbours are tried heaviest-edge first so a strong
 * bottleneck is found early and prunes hard.
 */
public final class BottleneckSolver {

    private final Graph g;
    private final BallCache balls;
    private final int nP;
    private final int[] pLabel;
    private final int[] pDeg;
    private final int[] parentOf;
    private final int[] anchorLabel;
    private final int[][] earlierV;
    private final int[][] earlierL;
    private final int[] map;
    private final boolean[] used;
    private double best;

    public BottleneckSolver(Graph g, DFSCode code, BallCache balls) {
        this.g = g;
        this.balls = balls;
        this.nP = code.nVertices();
        this.pLabel = new int[nP];
        this.pDeg = new int[nP];
        this.parentOf = new int[nP];
        this.anchorLabel = new int[nP];
        for (int v = 0; v < nP; v++) {
            pLabel[v] = code.vertexLabel(v);
            parentOf[v] = code.parentOf(v);
        }
        List<List<int[]>> earlier = new ArrayList<>();
        for (int v = 0; v < nP; v++) earlier.add(new ArrayList<>());
        for (EdgeTuple e : code.edges()) {
            int lo = Math.min(e.i(), e.j());
            int hi = Math.max(e.i(), e.j());
            earlier.get(hi).add(new int[] {lo, e.le()});
            pDeg[e.i()]++;
            pDeg[e.j()]++;
            if (e.isForward()) anchorLabel[e.j()] = e.le();
        }
        earlierV = new int[nP][];
        earlierL = new int[nP][];
        for (int v = 0; v < nP; v++) {
            List<int[]> list = earlier.get(v);
            earlierV[v] = new int[list.size()];
            earlierL[v] = new int[list.size()];
            for (int k = 0; k < list.size(); k++) {
                earlierV[v][k] = list.get(k)[0];
                earlierL[v][k] = list.get(k)[1];
            }
        }
        this.map = new int[nP];
        this.used = new boolean[g.n()];
    }

    /** @return B(P); 0.0 when the pattern does not occur (so u_flow is 0). */
    public double solve() {
        best = 0.0;
        for (int u0 = 0; u0 < g.n(); u0++) {
            if (g.vlabel(u0) != pLabel[0] || g.degree(u0) < pDeg[0]) continue;
            map[0] = u0;
            used[u0] = true;
            assign(1, Double.POSITIVE_INFINITY);
            used[u0] = false;
            map[0] = -1;
        }
        return best;
    }

    /** Extend the mapping to pattern vertex k; runningMin = min weight over edges placed so far. */
    private void assign(int k, double runningMin) {
        if (runningMin <= best) return;                 // branch-and-bound: cannot improve
        if (k == nP) {
            best = runningMin;                          // complete occurrence, strictly better
            return;
        }
        int au = map[parentOf[k]];
        int[] nbrs = g.neighbors(au);
        int[] labs = g.neighborEdgeLabels(au);
        for (int t : heaviestFirst(au)) {
            if (labs[t] != anchorLabel[k]) continue;
            int x = nbrs[t];
            if (used[x] || g.vlabel(x) != pLabel[k] || g.degree(x) < pDeg[k]) continue;
            if (!balls.inBall(map[0], x)) continue;     // MER constraint (d)
            double m = runningMin;
            boolean ok = true;
            for (int e = 0; e < earlierV[k].length; e++) {
                int pu = map[earlierV[k][e]];
                if (g.edgeLabelBetween(pu, x) != earlierL[k][e]) { ok = false; break; }  // absent or wrong label
                double w = g.edgeWeightBetween(pu, x);
                if (w < m) m = w;
            }
            if (!ok || m <= best) continue;             // prune before recursing
            map[k] = x;
            used[x] = true;
            assign(k + 1, m);
            used[x] = false;
            map[k] = -1;
        }
    }

    /** Neighbour slot indices of {@code au}, ordered by descending edge weight. */
    private int[] heaviestFirst(int au) {
        double[] w = g.neighborEdgeWeights(au);
        Integer[] order = new Integer[w.length];
        for (int i = 0; i < w.length; i++) order[i] = i;
        java.util.Arrays.sort(order, (a, b) -> Double.compare(w[b], w[a]));
        int[] out = new int[w.length];
        for (int i = 0; i < w.length; i++) out[i] = order[i];
        return out;
    }
}
