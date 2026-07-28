package edu.huflit.inchusm;

import edu.huflit.inchusm.csp.BallCache;
import edu.huflit.inchusm.csp.BottleneckSolver;
import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.mining.ValueCompute;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * B(P) = max over occurrences of min edge weight, checked two ways:
 *   (1) against a brute-force reference (enumerate every injective label-preserving
 *       occurrence, min-then-max) -- the Java branch-and-bound must equal it exactly;
 *   (2) anti-monotonicity: B(child) <= B(parent), the property that lets FLOW mine
 *       without an upper bound. A large MER is used so the ball never constrains, so
 *       the test measures the bottleneck logic, not locality.
 */
class BottleneckTest {

    /** Brute-force B(P): every injective, label-preserving, edge-preserving map P->G. */
    private static double referenceBottleneck(Graph g, DFSCode code) {
        int nP = code.nVertices();
        int[] plab = new int[nP];
        for (int v = 0; v < nP; v++) plab[v] = code.vertexLabel(v);
        List<int[]> pedges = new ArrayList<>();
        for (EdgeTuple e : code.edges()) pedges.add(new int[] {e.i(), e.j(), e.le()});
        int[] map = new int[nP];
        java.util.Arrays.fill(map, -1);
        boolean[] used = new boolean[g.n()];
        return bt(g, plab, pedges, map, used, 0, Double.NEGATIVE_INFINITY);
    }

    private static double bt(Graph g, int[] plab, List<int[]> pe, int[] map, boolean[] used,
                             int k, double best) {
        if (k == plab.length) {
            double m = Double.POSITIVE_INFINITY;
            for (int[] e : pe) m = Math.min(m, g.edgeWeightBetween(map[e[0]], map[e[1]]));
            return Math.max(best, m);
        }
        for (int u = 0; u < g.n(); u++) {
            if (used[u] || g.vlabel(u) != plab[k]) continue;
            boolean ok = true;
            for (int[] e : pe) {
                int a = e[0] == k ? u : (map[e[0]] >= 0 && e[0] < k ? map[e[0]] : -2);
                int b = e[1] == k ? u : (map[e[1]] >= 0 && e[1] < k ? map[e[1]] : -2);
                if (e[0] > k || e[1] > k) continue;          // edge not fully placed yet
                if (a == -2 || b == -2) continue;
                if (g.edgeLabelBetween(a, b) != e[2]) { ok = false; break; }
            }
            if (!ok) continue;
            map[k] = u;
            used[u] = true;
            best = bt(g, plab, pe, map, used, k + 1, best);
            used[u] = false;
            map[k] = -1;
        }
        return best;
    }

    private static Graph randomWeightedGraph(Random r, int n, int m, int nLabels) {
        int[] vlab = new int[n];
        for (int i = 0; i < n; i++) vlab[i] = r.nextInt(nLabels);
        List<int[]> edges = new ArrayList<>();
        List<Double> w = new ArrayList<>();
        java.util.Set<Long> seen = new java.util.HashSet<>();
        int guard = 0;
        while (edges.size() < m && guard++ < m * 20) {
            int u = r.nextInt(n), v = r.nextInt(n);
            if (u == v) continue;
            long key = (long) Math.min(u, v) * n + Math.max(u, v);
            if (!seen.add(key)) continue;
            edges.add(new int[] {u, v, r.nextInt(2)});           // edge label 0 or 1
            w.add(Math.round(Math.pow(10, r.nextDouble() * 3) * 100) / 100.0);  // heavy-tailed
        }
        double[] wa = new double[w.size()];
        for (int i = 0; i < wa.length; i++) wa[i] = w.get(i);
        return new Graph(vlab, edges, wa);
    }

    /** Every connected 1..maxE-edge pattern lifted from G, so it occurs at least once. */
    private static List<DFSCode> liftedPatterns(Graph g, Random r, int want, int maxE) {
        List<DFSCode> out = new ArrayList<>();
        List<int[]> gEdges = new ArrayList<>();
        for (int u = 0; u < g.n(); u++)
            for (int t = 0; t < g.degree(u); t++) {
                int v = g.neighbors(u)[t];
                if (u < v) gEdges.add(new int[] {u, v});
            }
        if (gEdges.isEmpty()) return out;
        for (int tries = 0; tries < want * 8 && out.size() < want; tries++) {
            int k = 1 + r.nextInt(maxE);
            java.util.LinkedHashSet<Integer> vs = new java.util.LinkedHashSet<>();
            List<int[]> chosen = new ArrayList<>();
            int[] s = gEdges.get(r.nextInt(gEdges.size()));
            chosen.add(s); vs.add(s[0]); vs.add(s[1]);
            for (int i = 1; i < k; i++) {
                List<int[]> cand = new ArrayList<>();
                for (int[] e : gEdges)
                    if (!chosen.contains(e) && (vs.contains(e[0]) || vs.contains(e[1]))) cand.add(e);
                if (cand.isEmpty()) break;
                int[] e = cand.get(r.nextInt(cand.size()));
                chosen.add(e); vs.add(e[0]); vs.add(e[1]);
            }
            // relabel to a small graph and take its min DFS code
            List<Integer> vl = new ArrayList<>(vs);
            int[] plab = new int[vl.size()];
            for (int i = 0; i < vl.size(); i++) plab[i] = g.vlabel(vl.get(i));
            List<int[]> pe = new ArrayList<>();
            for (int[] e : chosen)
                pe.add(new int[] {vl.indexOf(e[0]), vl.indexOf(e[1]),
                        g.edgeLabelBetween(e[0], e[1])});
            out.add(DFSCode.of(edu.huflit.inchusm.dfs.MinCode.minCodeOf(new Graph(plab, pe))));
        }
        return out;
    }

    @Test
    void matchesBruteForceReference() {
        Random r = new Random(20260725L);
        int checked = 0;
        for (int trial = 0; trial < 60; trial++) {
            Graph g = randomWeightedGraph(r, 6 + r.nextInt(6), 8 + r.nextInt(12), 3);
            BallCache balls = new BallCache(g, 9);                // MER large: never constrains
            for (DFSCode code : liftedPatterns(g, r, 8, 3)) {
                double got = new BottleneckSolver(g, code, balls).solve();
                double ref = referenceBottleneck(g, code);
                if (ref == Double.NEGATIVE_INFINITY) ref = 0.0;   // no occurrence
                assertEquals(ref, got, 1e-9, "B(P) mismatch for " + code.key());
                checked++;
            }
        }
        assertTrue(checked > 100, "too few patterns exercised: " + checked);
    }

    @Test
    void flowIsAntiMonotone() {
        Random r = new Random(424242L);
        int pairs = 0;
        for (int trial = 0; trial < 40; trial++) {
            Graph g = randomWeightedGraph(r, 8 + r.nextInt(6), 12 + r.nextInt(14), 4);
            BallCache balls = new BallCache(g, 9);
            for (DFSCode parent : liftedPatterns(g, r, 10, 3)) {
                ValueCompute.Evaluation pe = ValueCompute.evaluateFlow(g, parent, balls);
                if (pe.cmni() == 0) continue;
                for (EdgeTuple e : edu.huflit.inchusm.mining.Extensions.rightmost(
                        parent, new edu.huflit.inchusm.mining.Alphabet(g.distinctEdgeTypes()))) {
                    DFSCode child = parent.extend(e);
                    if (!edu.huflit.inchusm.dfs.MinCode.isMin(child)) continue;
                    ValueCompute.Evaluation ce = ValueCompute.evaluateFlow(g, child, balls);
                    assertTrue(ce.flow() <= pe.flow() + 1e-9,
                            "u_flow rose on extension: " + parent.key() + " -> " + child.key()
                                    + " (" + pe.flow() + " -> " + ce.flow() + ")");
                    pairs++;
                }
            }
        }
        assertTrue(pairs > 50, "too few parent-child pairs: " + pairs);
    }
}
