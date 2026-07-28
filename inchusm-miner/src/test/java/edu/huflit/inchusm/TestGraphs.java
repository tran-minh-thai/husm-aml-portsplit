package edu.huflit.inchusm;

import edu.huflit.inchusm.graph.Graph;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/** Deterministic random labeled graphs for equivalence tests. */
public final class TestGraphs {

    private TestGraphs() {
    }

    /** Connected graph: spanning tree + extra random non-parallel edges. */
    public static Graph random(long seed, int n, int extraEdges, int nVertexLabels, int nEdgeLabels) {
        Random r = new Random(seed);
        int[] vlabel = new int[n];
        for (int i = 0; i < n; i++) vlabel[i] = 1 + r.nextInt(nVertexLabels);

        List<int[]> edges = new ArrayList<>();
        Set<Long> present = new HashSet<>();
        for (int v = 1; v < n; v++) {
            int u = r.nextInt(v);
            edges.add(new int[] {u, v, r.nextInt(nEdgeLabels)});
            present.add(pair(u, v));
        }
        int attempts = 0;
        int added = 0;
        while (added < extraEdges && attempts < 200) {
            attempts++;
            int u = r.nextInt(n);
            int v = r.nextInt(n);
            if (u == v || present.contains(pair(u, v))) continue;
            edges.add(new int[] {u, v, r.nextInt(nEdgeLabels)});
            present.add(pair(u, v));
            added++;
        }
        return new Graph(vlabel, edges);
    }

    /** Random relabeling of vertex ids (labels travel with the vertices). */
    public static Graph permuted(Graph g, long seed) {
        Random r = new Random(seed);
        int n = g.n();
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        int[] vlabel = new int[n];
        for (int i = 0; i < n; i++) vlabel[perm[i]] = g.vlabel(i);
        List<int[]> edges = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            int[] nbrs = g.neighbors(u);
            int[] labs = g.neighborEdgeLabels(u);
            for (int k = 0; k < nbrs.length; k++) {
                if (u < nbrs[k]) edges.add(new int[] {perm[u], perm[nbrs[k]], labs[k]});
            }
        }
        return new Graph(vlabel, edges);
    }

    private static long pair(int a, int b) {
        int lo = Math.min(a, b);
        int hi = Math.max(a, b);
        return ((long) lo << 32) | hi;
    }

    /**
     * Disjoint union: concatenate the graphs offsetting vertex ids so the parts
     * share no vertex and no edge crosses between them  -  exactly the Elliptic
     * "sequence of disjoint snapshots" setting the incremental miner targets.
     */
    public static Graph disjointUnion(List<Graph> parts) {
        int total = 0;
        for (Graph p : parts) total += p.n();
        int[] vlabel = new int[total];
        List<int[]> edges = new ArrayList<>();
        int offset = 0;
        for (Graph p : parts) {
            for (int u = 0; u < p.n(); u++) vlabel[offset + u] = p.vlabel(u);
            for (int u = 0; u < p.n(); u++) {
                int[] nbrs = p.neighbors(u);
                int[] labs = p.neighborEdgeLabels(u);
                for (int k = 0; k < nbrs.length; k++) {
                    if (u < nbrs[k]) edges.add(new int[] {offset + u, offset + nbrs[k], labs[k]});
                }
            }
            offset += p.n();
        }
        return new Graph(vlabel, edges);
    }
}
