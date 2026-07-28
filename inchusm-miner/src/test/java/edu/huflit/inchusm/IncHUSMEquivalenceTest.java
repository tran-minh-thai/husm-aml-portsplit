package edu.huflit.inchusm;

import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.mining.HUSM;
import edu.huflit.inchusm.mining.IncHUSM;
import edu.huflit.inchusm.mining.Measure;
import edu.huflit.inchusm.mining.PatternState;
import edu.huflit.inchusm.mining.UtilityTable;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The core guarantee of the incremental miner (invariants INV1-INV3).
 * Stream disjoint snapshots one at a time; after EACH snapshot the incremental
 * store must equal a definition-level static computation on the accumulated
 * union  -  for both the utility and the support measure.
 */
class IncHUSMEquivalenceTest {

    private static final Map<Integer, Double> EXT = Map.of(1, 1.0, 2, 2.5, 3, 4.5);
    private static final int MAX_EDGES = 4;
    private static final int MER = 2;

    @Test
    void utilityMeasureMatchesStaticAfterEachSnapshot() {
        UtilityTable ext = new UtilityTable(EXT);
        for (long base : new long[] {10, 20, 30, 40, 50}) {
            List<Graph> stream = randomStream(base, 5);
            double minUtil = pickUtilityThreshold(stream, ext);

            IncHUSM inc = new IncHUSM(
                    new IncHUSM.Config(minUtil, MER, HUSM.Bound.CEU, Measure.UTILITY, MAX_EDGES), ext);

            Set<String> prevHigh = new HashSet<>();
            List<Graph> seen = new ArrayList<>();
            for (int t = 0; t < stream.size(); t++) {
                inc.update(stream.get(t));
                seen.add(stream.get(t));
                Graph union = TestGraphs.disjointUnion(seen);

                Map<String, Double> expected = BruteForce.hus(union, ext, MER, MAX_EDGES, minUtil);
                Set<String> got = highKeys(inc);
                String ctx = "base=" + base + " t=" + (t + 1) + " minUtil=" + minUtil;
                assertEquals(expected.keySet(), got, ctx);

                // INV1: stored value equals the definition-level union value
                for (PatternState st : inc.highPatterns()) {
                    assertEquals(expected.get(st.code().key()),
                            st.evaluation().utility(), 1e-9, ctx + " code=" + st.code().key());
                }
                // INV3: with an absolute threshold the high set only grows
                assertTrue(got.containsAll(prevHigh), ctx + " high set shrank (monotonicity broken)");
                prevHigh = got;
            }
        }
    }

    @Test
    void supportMeasureMatchesStaticAfterEachSnapshot() {
        UtilityTable ext = new UtilityTable(EXT);
        for (long base : new long[] {11, 22, 33, 44}) {
            List<Graph> stream = randomStream(base, 5);
            int minSup = 3;

            IncHUSM inc = new IncHUSM(
                    new IncHUSM.Config(minSup, MER, HUSM.Bound.CEU, Measure.SUPPORT, MAX_EDGES), ext);

            List<Graph> seen = new ArrayList<>();
            for (int t = 0; t < stream.size(); t++) {
                inc.update(stream.get(t));
                seen.add(stream.get(t));
                Graph union = TestGraphs.disjointUnion(seen);

                Set<String> expected = new TreeSet<>();
                for (var e : BruteForce.allPatternStats(union, ext, MER, MAX_EDGES).entrySet()) {
                    if (e.getValue().cmni() >= minSup) expected.add(e.getKey());
                }
                assertEquals(expected, highKeys(inc), "base=" + base + " t=" + (t + 1));
            }
        }
    }

    @Test
    void incrementalEqualsStaticHusmOnFinalUnion() {
        // cross-check against the OTHER static implementation (HUSM.mine), not just brute force
        UtilityTable ext = new UtilityTable(EXT);
        List<Graph> stream = randomStream(123, 6);
        double minUtil = pickUtilityThreshold(stream, ext);

        IncHUSM inc = new IncHUSM(
                new IncHUSM.Config(minUtil, MER, HUSM.Bound.CEU, Measure.UTILITY, MAX_EDGES), ext);
        for (Graph s : stream) inc.update(s);

        Graph union = TestGraphs.disjointUnion(stream);
        HUSM.Result stat = HUSM.mine(union, ext, new HUSM.Config(minUtil, MER, HUSM.Bound.CEU, MAX_EDGES));
        assertEquals(stat.hus.keySet(), highKeys(inc));
    }

    private static Set<String> highKeys(IncHUSM inc) {
        Set<String> out = new TreeSet<>();
        for (PatternState st : inc.highPatterns()) out.add(st.code().key());
        return out;
    }

    private static List<Graph> randomStream(long base, int count) {
        List<Graph> parts = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            parts.add(TestGraphs.random(base + i, 6 + (i % 3), 3, 3, 2));
        }
        return parts;
    }

    /** A threshold strictly between two distinct union utility values. */
    private static double pickUtilityThreshold(List<Graph> stream, UtilityTable ext) {
        Graph union = TestGraphs.disjointUnion(stream);
        TreeSet<Double> utilities = new TreeSet<>();
        for (var s : BruteForce.allPatternStats(union, ext, MER, MAX_EDGES).values()) {
            if (s.utility() > 0) utilities.add(s.utility());
        }
        List<Double> sorted = new ArrayList<>(utilities);
        int mid = Math.max(1, sorted.size() / 2);
        return (sorted.get(mid - 1) + sorted.get(mid)) / 2.0;
    }
}
