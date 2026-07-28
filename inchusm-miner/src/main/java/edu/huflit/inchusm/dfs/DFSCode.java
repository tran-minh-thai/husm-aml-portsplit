package edu.huflit.inchusm.dfs;

import edu.huflit.inchusm.graph.Graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * An immutable DFS code (sequence of {@link EdgeTuple}) plus derived structure:
 * vertex labels, DFS-tree parents and the rightmost path.
 *
 * Patterns are ALWAYS handled through their minimum DFS code (canonical label,
 * gSpan Thm. 1); {@code key()} is therefore usable as a map key for pattern
 * identity, and prefix relations on codes mirror ancestor relations in the
 * enumeration tree (HUSM Lemma 1).
 */
public final class DFSCode {

    private static final Pattern TUPLE =
            Pattern.compile("\\((-?\\d+),(-?\\d+),(-?\\d+),(-?\\d+),(-?\\d+)\\)");

    private final EdgeTuple[] edges;
    private final int nVertices;
    private final int[] vlabels;
    private final int[] parent;
    private final int[] rightmostPath; // rightmost vertex first, root (0) last

    private DFSCode(EdgeTuple[] edges) {
        this.edges = edges;
        int nv = 0;
        for (EdgeTuple e : edges) nv = Math.max(nv, Math.max(e.i(), e.j()) + 1);
        this.nVertices = nv;
        this.vlabels = new int[nv];
        this.parent = new int[nv];
        Arrays.fill(parent, -1);
        for (EdgeTuple e : edges) {
            vlabels[e.i()] = e.li();
            vlabels[e.j()] = e.lj();
            if (e.isForward()) parent[e.j()] = e.i();
        }
        List<Integer> path = new ArrayList<>();
        for (int v = nv - 1; v >= 0; v = parent[v]) {
            path.add(v);
            if (v == 0) break;
        }
        this.rightmostPath = new int[path.size()];
        for (int k = 0; k < path.size(); k++) rightmostPath[k] = path.get(k);
    }

    public static DFSCode of(List<EdgeTuple> list) {
        return new DFSCode(list.toArray(new EdgeTuple[0]));
    }

    /**
     * Inverse of {@link #key()}: rebuild a code from its canonical string, so a pattern
     * set mined in one run can be read back and matched against another graph.
     */
    public static DFSCode parse(String key) {
        Matcher m = TUPLE.matcher(key);
        List<EdgeTuple> tuples = new ArrayList<>();
        int consumed = 0;
        while (m.find()) {
            if (m.start() != consumed) break;   // a gap means the string is not a code
            consumed = m.end();
            tuples.add(new EdgeTuple(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)),
                    Integer.parseInt(m.group(3)), Integer.parseInt(m.group(4)),
                    Integer.parseInt(m.group(5))));
        }
        if (tuples.isEmpty() || consumed != key.length()) {
            throw new IllegalArgumentException("not a DFS code: " + key);
        }
        return of(tuples);
    }

    public static DFSCode single(EdgeTuple e) {
        return new DFSCode(new EdgeTuple[] {e});
    }

    public DFSCode extend(EdgeTuple e) {
        EdgeTuple[] longer = Arrays.copyOf(edges, edges.length + 1);
        longer[edges.length] = e;
        return new DFSCode(longer);
    }

    public int size() {
        return edges.length;
    }

    public EdgeTuple get(int k) {
        return edges[k];
    }

    public List<EdgeTuple> edges() {
        return List.of(edges);
    }

    public int nVertices() {
        return nVertices;
    }

    public int vertexLabel(int v) {
        return vlabels[v];
    }

    /** DFS-tree parent of v via forward edges; -1 for the root. */
    public int parentOf(int v) {
        return parent[v];
    }

    /** Rightmost path, rightmost vertex first, ending at vertex 0. */
    public int[] rightmostPath() {
        return rightmostPath.clone();
    }

    public int rightmostVertex() {
        return nVertices - 1;
    }

    public boolean containsEdge(int a, int b) {
        for (EdgeTuple e : edges) {
            if ((e.i() == a && e.j() == b) || (e.i() == b && e.j() == a)) return true;
        }
        return false;
    }

    /** The pattern graph this code describes (vertices 0..n-1 with labels). */
    public Graph toPatternGraph() {
        List<int[]> es = new ArrayList<>(edges.length);
        for (EdgeTuple e : edges) es.add(new int[] {e.i(), e.j(), e.le()});
        return new Graph(vlabels, es);
    }

    public String key() {
        return keyOf(List.of(edges));
    }

    /** Key of the parent code (last edge removed); empty string for a 1-edge code. */
    public String parentKey() {
        return keyOf(List.of(edges).subList(0, edges.length - 1));
    }

    public static String keyOf(List<EdgeTuple> tuples) {
        StringBuilder sb = new StringBuilder();
        for (EdgeTuple e : tuples) sb.append(e);
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof DFSCode c && Arrays.equals(edges, c.edges);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(edges);
    }

    @Override
    public String toString() {
        return key();
    }
}
