package edu.huflit.inchusm.dfs;

import edu.huflit.inchusm.graph.Graph;

import java.util.ArrayList;
import java.util.List;

/**
 * Constructive computation of the MINIMUM DFS code of a small pattern graph,
 * following gSpan's DFS-lexicographic order (Yan &amp; Han 2002, Defs. 2-3):
 *
 * at every step the smallest valid rightmost-path extension over ALL embeddings
 * (projections) of the current minimal prefix is appended. Ordering rules:
 * backward extensions before forward ones; among backward, smaller target j
 * first, then smaller edge label; among forward, deeper source i first, then
 * smaller (edge label, target vertex label).
 *
 * {@code isMin(code)}  -  the anti-duplicate prune of gSpan/HUSM (each pattern is
 * expanded only at its canonical node in the enumeration tree)  -  is implemented
 * as reconstruction + equality.
 */
public final class MinCode {

    private MinCode() {
    }

    private record Projection(int[] map, boolean[] mappedPg, boolean[] usedEdge) {

        Projection withBackward(int eid) {
            boolean[] used = usedEdge.clone();
            used[eid] = true;
            return new Projection(map, mappedPg, used);
        }

        Projection withForward(int codeVertex, int pgVertex, int eid) {
            int[] m = map.clone();
            m[codeVertex] = pgVertex;
            boolean[] mp = mappedPg.clone();
            mp[pgVertex] = true;
            boolean[] used = usedEdge.clone();
            used[eid] = true;
            return new Projection(m, mp, used);
        }
    }

    public static boolean isMin(DFSCode code) {
        return minCodeOf(code.toPatternGraph()).equals(code.edges());
    }

    public static List<EdgeTuple> minCodeOf(Graph pg) {
        int n = pg.n();
        int m = pg.m();
        if (m == 0) return List.of();

        // --- step 0: smallest 1-edge tuple over both orientations of every edge
        EdgeTuple first = null;
        for (int u = 0; u < n; u++) {
            int[] nbrs = pg.neighbors(u);
            int[] labs = pg.neighborEdgeLabels(u);
            for (int k = 0; k < nbrs.length; k++) {
                EdgeTuple cand = new EdgeTuple(0, 1, pg.vlabel(u), labs[k], pg.vlabel(nbrs[k]));
                if (first == null || cmpFirst(cand, first) < 0) first = cand;
            }
        }
        List<EdgeTuple> code = new ArrayList<>();
        code.add(first);

        List<Projection> projections = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            int[] nbrs = pg.neighbors(u);
            int[] labs = pg.neighborEdgeLabels(u);
            int[] eids = pg.neighborEdgeIds(u);
            for (int k = 0; k < nbrs.length; k++) {
                int w = nbrs[k];
                if (pg.vlabel(u) == first.li() && labs[k] == first.le() && pg.vlabel(w) == first.lj()) {
                    int[] map = new int[n];
                    java.util.Arrays.fill(map, -1);
                    map[0] = u;
                    map[1] = w;
                    boolean[] mapped = new boolean[n];
                    mapped[u] = true;
                    mapped[w] = true;
                    boolean[] used = new boolean[m];
                    used[eids[k]] = true;
                    projections.add(new Projection(map, mapped, used));
                }
            }
        }

        // --- grow one minimal edge at a time until all pattern edges are used
        while (code.size() < m) {
            DFSCode built = DFSCode.of(code);
            int[] rmp = built.rightmostPath(); // rmv..0
            int rmv = built.rightmostVertex();
            int nb = built.nVertices();

            EdgeTuple ext = null;
            List<Projection> next = null;

            // 1) backward from rmv: smaller j first (iterate rmp from the root end)
            for (int idx = rmp.length - 1; idx >= 0 && ext == null; idx--) {
                int j = rmp[idx];
                if (j == rmv || built.containsEdge(rmv, j)) continue;
                int bestLe = Integer.MAX_VALUE;
                List<Projection> sup = new ArrayList<>();
                for (Projection p : projections) {
                    int pu = p.map()[rmv];
                    int pv = p.map()[j];
                    int le = pg.edgeLabelBetween(pu, pv);
                    if (le == Integer.MIN_VALUE) continue;
                    int eid = pg.edgeIdBetween(pu, pv);
                    if (p.usedEdge()[eid]) continue;
                    if (le < bestLe) {
                        bestLe = le;
                        sup.clear();
                    }
                    if (le == bestLe) sup.add(p.withBackward(eid));
                }
                if (bestLe != Integer.MAX_VALUE) {
                    ext = new EdgeTuple(rmv, j, built.vertexLabel(rmv), bestLe, built.vertexLabel(j));
                    next = sup;
                }
            }

            // 2) forward from rightmost-path vertices: deeper i first (rmp is rmv..0)
            if (ext == null) {
                for (int idx = 0; idx < rmp.length && ext == null; idx++) {
                    int i = rmp[idx];
                    int bestLe = Integer.MAX_VALUE;
                    int bestLx = Integer.MAX_VALUE;
                    List<Projection> sup = new ArrayList<>();
                    for (Projection p : projections) {
                        int pu = p.map()[i];
                        int[] nbrs = pg.neighbors(pu);
                        int[] labs = pg.neighborEdgeLabels(pu);
                        int[] eids = pg.neighborEdgeIds(pu);
                        for (int k = 0; k < nbrs.length; k++) {
                            int x = nbrs[k];
                            if (p.mappedPg()[x]) continue;
                            int le = labs[k];
                            int lx = pg.vlabel(x);
                            if (le < bestLe || (le == bestLe && lx < bestLx)) {
                                bestLe = le;
                                bestLx = lx;
                                sup.clear();
                            }
                            if (le == bestLe && lx == bestLx) {
                                sup.add(p.withForward(nb, x, eids[k]));
                            }
                        }
                    }
                    if (bestLe != Integer.MAX_VALUE) {
                        ext = new EdgeTuple(i, nb, built.vertexLabel(i), bestLe, bestLx);
                        next = sup;
                    }
                }
            }

            if (ext == null) {
                throw new IllegalStateException("pattern graph is not connected");
            }
            code.add(ext);
            projections = next;
        }
        return code;
    }

    private static int cmpFirst(EdgeTuple a, EdgeTuple b) {
        if (a.li() != b.li()) return Integer.compare(a.li(), b.li());
        if (a.le() != b.le()) return Integer.compare(a.le(), b.le());
        return Integer.compare(a.lj(), b.lj());
    }
}
