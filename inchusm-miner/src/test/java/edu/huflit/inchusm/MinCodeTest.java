package edu.huflit.inchusm;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.dfs.MinCode;
import edu.huflit.inchusm.graph.Graph;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MinCodeTest {

    @Test
    void singleEdgeOrientation() {
        Graph pg = new Graph(new int[] {1, 2}, List.of(new int[] {0, 1, 7}));
        assertEquals(List.of(new EdgeTuple(0, 1, 1, 7, 2)), MinCode.minCodeOf(pg));

        assertTrue(MinCode.isMin(DFSCode.single(new EdgeTuple(0, 1, 1, 7, 2))));
        assertFalse(MinCode.isMin(DFSCode.single(new EdgeTuple(0, 1, 2, 7, 1))));
    }

    @Test
    void pathThreeVertices() {
        // labels 1-2-3, edge labels 0
        Graph pg = new Graph(new int[] {1, 2, 3},
                List.of(new int[] {0, 1, 0}, new int[] {1, 2, 0}));
        assertEquals(
                List.of(new EdgeTuple(0, 1, 1, 0, 2), new EdgeTuple(1, 2, 2, 0, 3)),
                MinCode.minCodeOf(pg));

        // valid alternative DFS code rooted at the middle vertex must be non-minimal
        DFSCode alt = DFSCode.of(List.of(
                new EdgeTuple(0, 1, 2, 0, 1), new EdgeTuple(0, 2, 2, 0, 3)));
        assertFalse(MinCode.isMin(alt));
    }

    @Test
    void triangleUniformLabels() {
        Graph pg = new Graph(new int[] {1, 1, 1},
                List.of(new int[] {0, 1, 0}, new int[] {1, 2, 0}, new int[] {2, 0, 0}));
        // grow two forward edges then close the cycle with the backward edge (2,0)
        assertEquals(
                List.of(new EdgeTuple(0, 1, 1, 0, 1),
                        new EdgeTuple(1, 2, 1, 0, 1),
                        new EdgeTuple(2, 0, 1, 0, 1)),
                MinCode.minCodeOf(pg));
    }

    @Test
    void minimalCodeIsInvariantUnderVertexPermutation() {
        for (long seed = 1; seed <= 50; seed++) {
            Graph pg = TestGraphs.random(seed, 6, 3, 3, 2);
            List<EdgeTuple> a = MinCode.minCodeOf(pg);
            List<EdgeTuple> b = MinCode.minCodeOf(TestGraphs.permuted(pg, seed * 31 + 7));
            assertEquals(a, b, "seed " + seed);
        }
    }
}
