package edu.huflit.inchusm;

import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.mining.HUSM;
import edu.huflit.inchusm.mining.IncHUSM;
import edu.huflit.inchusm.mining.Measure;
import edu.huflit.inchusm.mining.PatternState;
import edu.huflit.inchusm.mining.UtilityTable;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Sliding-window IncHUSM: the high set must equal static mining on the WINDOW union.
 *
 * The add-only miner accumulates forever; here the oldest snapshot is evicted after each
 * update, so the answer tracks a moving window. This is the exactness claim that the
 * eviction path has to satisfy: because Lemma L1 makes a pattern's value the combination
 * of exactly the segments inside the window, removing a snapshot's segment must leave the
 * stored value equal to a definition-level static computation on the window union.
 *
 * Unlike the add-only case the high set is NOT monotone: a pattern leaves it when the
 * snapshot carrying its value falls out. The last test asserts that this actually happens,
 * so the window is doing real work rather than silently behaving add-only.
 */
class SlidingWindowTest {

    private static final Map<Integer, Double> EXT = Map.of(1, 1.0, 2, 2.5, 3, 4.5);
    private static final int MAX_EDGES = 4;
    private static final int MER = 2;

    @Test
    void windowedHighSetMatchesStaticOnWindowUnion() {
        UtilityTable ext = new UtilityTable(EXT);
        for (int window : new int[] {1, 2, 3}) {
            for (long base : new long[] {10, 20, 30}) {
                List<Graph> stream = randomStream(base, 6);
                double minUtil = pickUtilityThreshold(stream, ext);

                IncHUSM inc = new IncHUSM(
                        new IncHUSM.Config(minUtil, MER, HUSM.Bound.CEU, Measure.UTILITY, MAX_EDGES),
                        ext, null, window);

                for (int t = 0; t < stream.size(); t++) {
                    inc.update(stream.get(t));

                    List<Graph> inWindow = windowOf(stream, t, window);
                    Graph union = TestGraphs.disjointUnion(inWindow);
                    Map<String, Double> expected =
                            BruteForce.hus(union, ext, MER, MAX_EDGES, minUtil);

                    String ctx = "window=" + window + " base=" + base + " t=" + (t + 1);
                    assertEquals(expected.keySet(), highKeys(inc), ctx);

                    // every reported value must equal the definition-level window value
                    for (PatternState st : inc.highPatterns()) {
                        assertEquals(expected.get(st.code().key()),
                                st.evaluation().utility(), 1e-9, ctx + " code=" + st.code().key());
                    }
                }
            }
        }
    }

    @Test
    void windowOfOneEqualsStaticOnEachSnapshotAlone() {
        UtilityTable ext = new UtilityTable(EXT);
        List<Graph> stream = randomStream(77, 5);
        double minUtil = pickUtilityThreshold(stream, ext);

        IncHUSM inc = new IncHUSM(
                new IncHUSM.Config(minUtil, MER, HUSM.Bound.CEU, Measure.UTILITY, MAX_EDGES),
                ext, null, 1);

        for (int t = 0; t < stream.size(); t++) {
            inc.update(stream.get(t));
            HUSM.Result stat = HUSM.mine(stream.get(t), ext,
                    new HUSM.Config(minUtil, MER, HUSM.Bound.CEU, MAX_EDGES));
            assertEquals(stat.hus.keySet(), highKeys(inc), "t=" + (t + 1));
        }
    }

    @Test
    void stateStaysBoundedAndHighSetCanShrink() {
        UtilityTable ext = new UtilityTable(EXT);
        List<Graph> stream = randomStream(31, 8);
        double minUtil = pickUtilityThreshold(stream, ext);

        IncHUSM windowed = new IncHUSM(
                new IncHUSM.Config(minUtil, MER, HUSM.Bound.CEU, Measure.UTILITY, MAX_EDGES),
                ext, null, 2);
        IncHUSM addOnly = new IncHUSM(
                new IncHUSM.Config(minUtil, MER, HUSM.Bound.CEU, Measure.UTILITY, MAX_EDGES), ext);

        boolean shrankOnce = false;
        Set<String> prev = new TreeSet<>();
        for (Graph s : stream) {
            windowed.update(s);
            addOnly.update(s);
            Set<String> now = highKeys(windowed);
            if (!now.containsAll(prev)) shrankOnce = true;   // a pattern left the high set
            prev = now;
        }
        assertTrue(shrankOnce,
                "the windowed high set never shrank; the window is not evicting anything");
        assertTrue(windowed.stats().patternsTracked <= addOnly.stats().patternsTracked,
                "windowed state should not exceed the add-only state");
    }

    private static List<Graph> windowOf(List<Graph> stream, int t, int window) {
        int from = Math.max(0, t - window + 1);
        return new ArrayList<>(stream.subList(from, t + 1));
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
