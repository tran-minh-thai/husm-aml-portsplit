package edu.huflit.inchusm;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.dfs.MinCode;
import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.mining.UtilityTable;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Independent reference implementation for equivalence tests (E1b of the
 * experiment plan): enumerate ALL connected edge subsets of the data graph up to
 * maxEdges, canonicalize each induced pattern, enumerate ALL occurrences by
 * naive backtracking (no CSP shortcuts, no marking), filter by MER using its own
 * BFS, and compute CMNI + utility straight from the definitions.
 *
 * Deliberately shares only {@link MinCode#minCodeOf} with production code (its
 * correctness is covered by dedicated unit + permutation-invariance tests).
 */
public final class BruteForce {

    public record PatternStat(int cmni, double utility) {
    }

    private BruteForce() {
    }

    public static Map<String, PatternStat> allPatternStats(Graph g, UtilityTable ext,
                                                           int mer, int maxEdges) {
        // --- collect edges as {u, v, label} indexed by edge id
        int[][] edgeArr = new int[g.m()][];
        for (int u = 0; u < g.n(); u++) {
            int[] nbrs = g.neighbors(u);
            int[] labs = g.neighborEdgeLabels(u);
            int[] eids = g.neighborEdgeIds(u);
            for (int k = 0; k < nbrs.length; k++) {
                if (u < nbrs[k]) edgeArr[eids[k]] = new int[] {u, nbrs[k], labs[k]};
            }
        }

        // --- enumerate connected edge subsets (grow by adjacency, dedupe by BitSet)
        Set<BitSet> seen = new HashSet<>();
        for (int e = 0; e < g.m(); e++) {
            BitSet start = new BitSet(g.m());
            start.set(e);
            expand(g, edgeArr, start, maxEdges, seen);
        }

        // --- canonicalize and evaluate each distinct pattern once
        Map<String, PatternStat> stats = new TreeMap<>();
        Set<String> done = new HashSet<>();
        for (BitSet subset : seen) {
            DFSCode code = canonicalPattern(g, edgeArr, subset);
            String key = code.key();
            if (!done.add(key)) continue;
            stats.put(key, evaluatePattern(g, ext, code, mer));
        }
        return stats;
    }

    public static Map<String, Double> hus(Graph g, UtilityTable ext, int mer,
                                          int maxEdges, double minUtil) {
        Map<String, Double> out = new TreeMap<>();
        for (Map.Entry<String, PatternStat> e : allPatternStats(g, ext, mer, maxEdges).entrySet()) {
            if (e.getValue().utility() >= minUtil) out.put(e.getKey(), e.getValue().utility());
        }
        return out;
    }

    private static void expand(Graph g, int[][] edgeArr, BitSet current, int maxEdges,
                               Set<BitSet> seen) {
        if (!seen.add((BitSet) current.clone())) return;
        if (current.cardinality() >= maxEdges) return;
        // edges adjacent to the current vertex set
        Set<Integer> vertices = new HashSet<>();
        for (int e = current.nextSetBit(0); e >= 0; e = current.nextSetBit(e + 1)) {
            vertices.add(edgeArr[e][0]);
            vertices.add(edgeArr[e][1]);
        }
        for (int v : vertices) {
            for (int eid : g.neighborEdgeIds(v)) {
                if (!current.get(eid)) {
                    current.set(eid);
                    expand(g, edgeArr, current, maxEdges, seen);
                    current.clear(eid);
                }
            }
        }
    }

    private static DFSCode canonicalPattern(Graph g, int[][] edgeArr, BitSet subset) {
        List<Integer> vs = new ArrayList<>();
        Set<Integer> vset = new HashSet<>();
        for (int e = subset.nextSetBit(0); e >= 0; e = subset.nextSetBit(e + 1)) {
            for (int end = 0; end < 2; end++) {
                if (vset.add(edgeArr[e][end])) vs.add(edgeArr[e][end]);
            }
        }
        vs.sort(null);
        Map<Integer, Integer> renum = new HashMap<>();
        int[] vlabel = new int[vs.size()];
        for (int i = 0; i < vs.size(); i++) {
            renum.put(vs.get(i), i);
            vlabel[i] = g.vlabel(vs.get(i));
        }
        List<int[]> pedges = new ArrayList<>();
        for (int e = subset.nextSetBit(0); e >= 0; e = subset.nextSetBit(e + 1)) {
            pedges.add(new int[] {renum.get(edgeArr[e][0]), renum.get(edgeArr[e][1]), edgeArr[e][2]});
        }
        return DFSCode.of(MinCode.minCodeOf(new Graph(vlabel, pedges)));
    }

    /** Occurrence enumeration + CMNI + utility straight from Defs. 7-8. */
    private static PatternStat evaluatePattern(Graph g, UtilityTable ext, DFSCode code, int mer) {
        int nP = code.nVertices();
        int[] pLabel = new int[nP];
        for (int v = 0; v < nP; v++) pLabel[v] = code.vertexLabel(v);
        List<int[]> pEdges = new ArrayList<>();
        for (EdgeTuple e : code.edges()) pEdges.add(new int[] {e.i(), e.j(), e.le()});

        List<int[]> occurrences = new ArrayList<>();
        int[] map = new int[nP];
        Arrays.fill(map, -1);
        boolean[] used = new boolean[g.n()];
        enumerate(g, pLabel, pEdges, 0, map, used, occurrences);

        // MER filter with an independent BFS from each occurrence's f(I0)
        List<Set<Integer>> images = new ArrayList<>();
        for (int v = 0; v < nP; v++) images.add(new HashSet<>());
        Map<Integer, int[]> distCache = new HashMap<>();
        for (int[] occ : occurrences) {
            int[] dist = distCache.computeIfAbsent(occ[0], c -> bfs(g, c));
            boolean valid = true;
            for (int v = 0; v < nP; v++) {
                if (dist[occ[v]] < 0 || dist[occ[v]] > mer) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                for (int v = 0; v < nP; v++) images.get(v).add(occ[v]);
            }
        }
        int cmni = Integer.MAX_VALUE;
        for (Set<Integer> col : images) cmni = Math.min(cmni, col.size());
        if (cmni == 0 || cmni == Integer.MAX_VALUE) return new PatternStat(0, 0);

        double mean = 0;
        for (int v = 0; v < nP; v++) mean += ext.of(pLabel[v]);
        mean /= nP;
        return new PatternStat(cmni, cmni * mean);
    }

    private static void enumerate(Graph g, int[] pLabel, List<int[]> pEdges, int k,
                                  int[] map, boolean[] used, List<int[]> out) {
        if (k == pLabel.length) {
            out.add(map.clone());
            return;
        }
        for (int x = 0; x < g.n(); x++) {
            if (used[x] || g.vlabel(x) != pLabel[k]) continue;
            boolean ok = true;
            for (int[] e : pEdges) {
                int a = e[0];
                int b = e[1];
                int other = a == k ? b : (b == k ? a : -1);
                if (other >= 0 && other < k && g.edgeLabelBetween(map[other], x) != e[2]) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            map[k] = x;
            used[x] = true;
            enumerate(g, pLabel, pEdges, k + 1, map, used, out);
            used[x] = false;
            map[k] = -1;
        }
    }

    private static int[] bfs(Graph g, int src) {
        int[] dist = new int[g.n()];
        Arrays.fill(dist, -1);
        dist[src] = 0;
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(src);
        while (!q.isEmpty()) {
            int u = q.poll();
            for (int v : g.neighbors(u)) {
                if (dist[v] < 0) {
                    dist[v] = dist[u] + 1;
                    q.add(v);
                }
            }
        }
        return dist;
    }
}
