package edu.huflit.inchusm.csp;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.graph.Graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * CSP model of "pattern P into data graph G" (HUSM Def. 10, after GraMi Def. 5)
 * with the extra constraint (d): every image lies within distance MER of the
 * image of the initial vertex I0 (= pattern vertex 0 of the minimum DFS code).
 *
 * Builds the exact CMNI table: for each column v and candidate u, decide whether
 * SOME MER-valid occurrence maps v -> u. GraMi's cross-marking reuse applies:
 * each found solution validates one image in EVERY column at once, so the number
 * of backtracking searches is far below (columns x domain sizes). Unlike GraMi we
 * cannot early-stop at a support threshold: HUSM needs the exact CMNI value
 * because it appears in the utility itself.
 */
public final class CSPSolver {

    private final Graph g;
    private final int mer;
    private final BallCache balls;
    private final int nP;
    private final int[] pLabel;
    private final int[] pDeg;
    private final int[] parentOf;
    private final int[] anchorLabel;   // edge label of the forward edge (parent, k)
    private final int[][] earlierV;    // per k: earlier endpoints of ALL pattern edges at k
    private final int[][] earlierL;    // per k: matching edge labels
    private final int[] map;
    private final boolean[] used;

    public CSPSolver(Graph g, DFSCode code, BallCache balls) {
        this.g = g;
        this.mer = balls.radius();
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
        Arrays.fill(map, -1);
        this.used = new boolean[g.n()];
    }

    public CMNITable solve() {
        List<Set<Integer>> valid = new ArrayList<>(nP);
        for (int v = 0; v < nP; v++) valid.add(new HashSet<>());

        for (int v = 0; v < nP; v++) {
            for (int u = 0; u < g.n(); u++) {
                if (g.vlabel(u) != pLabel[v] || g.degree(u) < pDeg[v]) continue;
                if (valid.get(v).contains(u)) continue;
                if (exists(v, u)) {
                    for (int w = 0; w < nP; w++) valid.get(w).add(map[w]);
                    unwind();
                }
            }
        }
        return new CMNITable(valid);
    }

    /** Is there a MER-valid occurrence with pattern vertex pinV mapped to pinU? */
    private boolean exists(int pinV, int pinU) {
        return assign(0, pinV, pinU);
    }

    private void unwind() {
        for (int k = 0; k < nP; k++) {
            if (map[k] >= 0) {
                used[map[k]] = false;
                map[k] = -1;
            }
        }
    }

    private boolean assign(int k, int pinV, int pinU) {
        if (k == nP) return true;
        if (k == pinV) {
            return tryOne(k, pinU, pinV, pinU);
        }
        if (k == 0) {
            // pinV > 0: x0 must lie within MER of pinU (distance is symmetric)
            for (int x : balls.ball(pinU)) {
                if (tryOne(0, x, pinV, pinU)) return true;
            }
            return false;
        }
        // candidates = neighbors of the image of k's DFS-tree parent via the forward edge
        int au = map[parentOf[k]];
        int[] nbrs = g.neighbors(au);
        int[] labs = g.neighborEdgeLabels(au);
        for (int t = 0; t < nbrs.length; t++) {
            if (labs[t] != anchorLabel[k]) continue;
            if (tryOne(k, nbrs[t], pinV, pinU)) return true;
        }
        return false;
    }

    private boolean tryOne(int k, int x, int pinV, int pinU) {
        if (used[x] || g.vlabel(x) != pLabel[k] || g.degree(x) < pDeg[k]) return false;
        if (k > 0 && !balls.inBall(map[0], x)) return false; // MER constraint (d)
        for (int e = 0; e < earlierV[k].length; e++) {
            if (g.edgeLabelBetween(map[earlierV[k][e]], x) != earlierL[k][e]) return false;
        }
        map[k] = x;
        used[x] = true;
        if (assign(k + 1, pinV, pinU)) return true; // keep assignment for the caller to read
        used[x] = false;
        map[k] = -1;
        return false;
    }
}
