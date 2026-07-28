package edu.huflit.inchusm;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.mining.BHUS;
import edu.huflit.inchusm.mining.BHUSInc;
import edu.huflit.inchusm.mining.HUSM.PatternResult;
import edu.huflit.inchusm.mining.ValueCompute.Evaluation;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * BHUSInc maintains the same Max/Min boundaries as the static {@link BHUS#extract},
 * regardless of the order in which patterns become high (a child may become high before
 * its parent, since utility is not monotone on the enumeration tree).
 */
class BHUSIncTest {

    private static PatternResult pr(DFSCode c) {
        return new PatternResult(c, new Evaluation(1, 1, 1, 1, 1));
    }

    /** Absorb the codes one snapshot at a time in the given order, compare to static extract. */
    private static void assertMatchesStatic(List<DFSCode> order) {
        BHUSInc inc = new BHUSInc();
        for (DFSCode c : order) inc.absorb(List.of(c));
        BHUS.Sets got = inc.snapshot();
        BHUS.Sets exp = BHUS.extract(order.stream().map(BHUSIncTest::pr).toList());
        assertEquals(exp.maxBoundary(), got.maxBoundary(), "maxBHUS");
        assertEquals(exp.minBoundary(), got.minBoundary(), "minBHUS");
    }

    @Test
    void chainAnyOrderMatchesStatic() {
        DFSCode c1 = DFSCode.single(new EdgeTuple(0, 1, 1, 0, 1));
        DFSCode c2 = c1.extend(new EdgeTuple(1, 2, 1, 0, 2));
        DFSCode c3 = c2.extend(new EdgeTuple(2, 3, 2, 0, 3));
        assertMatchesStatic(List.of(c1, c2, c3));   // parent-first
        assertMatchesStatic(List.of(c3, c2, c1));   // child-first (broken-path arrival)
        assertMatchesStatic(List.of(c2, c1, c3));   // shuffled
    }

    @Test
    void gapAnyOrderMatchesStatic() {
        DFSCode c1 = DFSCode.single(new EdgeTuple(0, 1, 1, 0, 1));
        DFSCode c2 = c1.extend(new EdgeTuple(1, 2, 1, 0, 2));
        DFSCode c3 = c2.extend(new EdgeTuple(2, 3, 2, 0, 3));
        assertMatchesStatic(List.of(c1, c3));       // c2 missing -> Min boundary at c3
        assertMatchesStatic(List.of(c3, c1));       // reversed
    }

    @Test
    void branchingChildrenBeforeParentMatchesStatic() {
        DFSCode c1 = DFSCode.single(new EdgeTuple(0, 1, 1, 0, 1));
        DFSCode a = c1.extend(new EdgeTuple(1, 2, 1, 0, 2));
        DFSCode b = c1.extend(new EdgeTuple(1, 2, 1, 0, 3));   // sibling of a
        assertMatchesStatic(List.of(c1, a, b));
        assertMatchesStatic(List.of(a, b, c1));                // both children before the parent
    }
}
