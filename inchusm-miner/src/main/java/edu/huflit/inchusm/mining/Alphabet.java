package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.graph.Graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The set of edge types allowed as rightmost-path extensions, plus a fast
 * (labelLo,labelHi) -> edge-labels index for backward extensions.
 *
 * HUSM uses the RETAINED 1-edges (those whose bound reaches the threshold  -  a
 * safe prune by downward closure). IncHUSM uses ALL edge types present in the
 * arriving snapshot (pruning the alphabet is an optimisation, not needed for
 * correctness). Shared so the two miners generate identical child sets.
 */
public final class Alphabet {

    private final List<Graph.EdgeType> forward;
    private final Map<Long, int[]> backwardLabelsByPair;

    public Alphabet(List<Graph.EdgeType> types) {
        this.forward = List.copyOf(types);
        Map<Long, List<Integer>> tmp = new HashMap<>();
        for (Graph.EdgeType t : types) {
            tmp.computeIfAbsent(pairKey(t.la(), t.lb()), k -> new ArrayList<>()).add(t.le());
        }
        this.backwardLabelsByPair = new HashMap<>();
        for (Map.Entry<Long, List<Integer>> e : tmp.entrySet()) {
            int[] arr = new int[e.getValue().size()];
            for (int i = 0; i < arr.length; i++) arr[i] = e.getValue().get(i);
            backwardLabelsByPair.put(e.getKey(), arr);
        }
    }

    public List<Graph.EdgeType> forwardTypes() {
        return forward;
    }

    /** Edge labels connecting a vertex labelled {@code a} to one labelled {@code b}. */
    public int[] backwardEdgeLabels(int a, int b) {
        int[] les = backwardLabelsByPair.get(pairKey(Math.min(a, b), Math.max(a, b)));
        return les == null ? EMPTY : les;
    }

    private static final int[] EMPTY = new int[0];

    private static long pairKey(int lo, int hi) {
        return ((long) lo << 32) | (hi & 0xffffffffL);
    }
}
