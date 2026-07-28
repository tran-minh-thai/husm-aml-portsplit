package edu.huflit.inchusm;

import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.mining.HUSM;
import edu.huflit.inchusm.mining.UtilityTable;
import edu.huflit.inchusm.mining.ValueCompute.Evaluation;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * E1b: HUSM must return EXACTLY the high-utility subgraphs found by definition-level
 * brute force, for every bound (bounds may only change how much is pruned, never
 * the result), plus the bound-dominance and candidate-count-ordering properties.
 */
class HUSMEquivalenceTest {

    private static final Map<Integer, Double> EXT =
            Map.of(1, 1.0, 2, 2.5, 3, 4.5);
    private static final int MAX_EDGES = 4;

    @Test
    void matchesBruteForceForAllBounds() {
        int combosChecked = 0;
        for (long seed : new long[] {1, 2, 3, 4, 5}) {
            Graph g = TestGraphs.random(seed, 9, 4, 3, 2);
            UtilityTable ext = new UtilityTable(EXT);
            for (int mer : new int[] {1, 2}) {
                var stats = BruteForce.allPatternStats(g, ext, mer, MAX_EDGES);
                TreeSet<Double> utilities = new TreeSet<>();
                for (var s : stats.values()) {
                    if (s.utility() > 0) utilities.add(s.utility());
                }
                if (utilities.size() < 2) continue;
                for (double minUtil : thresholds(utilities)) {
                    Map<String, Double> expected = BruteForce.hus(g, ext, mer, MAX_EDGES, minUtil);
                    long candIvu = -1;
                    long candEu = -1;
                    long candCeu = -1;
                    for (HUSM.Bound b : HUSM.Bound.values()) {
                        HUSM.Result res = HUSM.mine(g, ext,
                                new HUSM.Config(minUtil, mer, b, MAX_EDGES));
                        String ctx = "seed=" + seed + " mer=" + mer
                                + " minUtil=" + minUtil + " bound=" + b;
                        assertEquals(expected.keySet(), res.hus.keySet(), ctx);
                        for (var e : res.hus.entrySet()) {
                            assertEquals(expected.get(e.getKey()),
                                    e.getValue().eval().utility(), 1e-9, ctx);
                        }
                        switch (b) {
                            case IVU -> candIvu = res.stats.candidates;
                            case EU -> candEu = res.stats.candidates;
                            case CEU -> candCeu = res.stats.candidates;
                        }
                    }
                    assertTrue(candIvu >= candEu && candEu >= candCeu,
                            "candidate ordering IVU>=EU>=CEU violated: "
                                    + candIvu + "/" + candEu + "/" + candCeu);
                    combosChecked++;
                }
            }
        }
        assertTrue(combosChecked >= 10,
                "equivalence loop barely ran (" + combosChecked + " combos)  -  thresholds degenerate?");
    }

    @Test
    void boundsDominateUtility() {
        Graph g = TestGraphs.random(11, 10, 5, 3, 2);
        UtilityTable ext = new UtilityTable(EXT);
        HUSM.Result res = HUSM.mine(g, ext,
                new HUSM.Config(0.5, 2, HUSM.Bound.IVU, MAX_EDGES, true));
        assertTrue(res.allEvaluations.size() > 10);
        for (Evaluation ev : res.allEvaluations) {
            assertTrue(ev.utility() <= ev.ceu() + 1e-9, "u <= CEU violated: " + ev);
            assertTrue(ev.ceu() <= ev.eu() + 1e-9, "CEU <= EU violated: " + ev);
            assertTrue(ev.eu() <= ev.ivu() + 1e-9, "EU <= IVU violated: " + ev);
        }
    }

    /** A few thresholds strictly between consecutive distinct utility values. */
    private static List<Double> thresholds(TreeSet<Double> utilities) {
        List<Double> sorted = new ArrayList<>(utilities);
        List<Double> out = new ArrayList<>();
        int[] positions = {sorted.size() / 2, sorted.size() - 1};
        for (int pos : positions) {
            if (pos >= 1) out.add((sorted.get(pos - 1) + sorted.get(pos)) / 2.0);
        }
        return out;
    }
}
