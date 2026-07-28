package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.graph.Graph;

import java.util.LinkedHashSet;

/**
 * gSpan rightmost-path extension generation (Yan &amp; Han A.5), shared by HUSM
 * and IncHUSM so both explore an identical child set.
 *
 * Backward edges grow only from the rightmost vertex to a rightmost-path vertex;
 * forward edges grow a new vertex from any rightmost-path vertex. Edge labels and
 * neighbour vertex labels come from the {@link Alphabet}. The caller still applies
 * the {@code MinCode.isMin} anti-duplicate prune on each child.
 */
public final class Extensions {

    private Extensions() {
    }

    public static LinkedHashSet<EdgeTuple> rightmost(DFSCode code, Alphabet alpha) {
        LinkedHashSet<EdgeTuple> out = new LinkedHashSet<>();
        int[] rmp = code.rightmostPath();
        int rmv = code.rightmostVertex();
        int nb = code.nVertices();

        // backward: rightmost vertex -> a rightmost-path vertex not yet adjacent
        for (int idx = rmp.length - 1; idx >= 0; idx--) {
            int w = rmp[idx];
            if (w == rmv || code.containsEdge(rmv, w)) continue;
            for (int le : alpha.backwardEdgeLabels(code.vertexLabel(rmv), code.vertexLabel(w))) {
                out.add(new EdgeTuple(rmv, w, code.vertexLabel(rmv), le, code.vertexLabel(w)));
            }
        }
        // forward: any rightmost-path vertex -> a new vertex nb
        for (int v : rmp) {
            int lv = code.vertexLabel(v);
            for (Graph.EdgeType t : alpha.forwardTypes()) {
                if (t.la() == lv) out.add(new EdgeTuple(v, nb, lv, t.le(), t.lb()));
                if (t.lb() == lv && t.la() != t.lb()) {
                    out.add(new EdgeTuple(v, nb, lv, t.le(), t.la()));
                }
            }
        }
        return out;
    }
}
