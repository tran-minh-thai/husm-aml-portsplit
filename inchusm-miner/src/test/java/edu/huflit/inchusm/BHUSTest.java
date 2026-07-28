package edu.huflit.inchusm;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.mining.BHUS;
import edu.huflit.inchusm.mining.HUSM.PatternResult;
import edu.huflit.inchusm.mining.ValueCompute.Evaluation;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BHUSTest {

    private static PatternResult pr(DFSCode c) {
        return new PatternResult(c, new Evaluation(1, 1, 1, 1, 1));
    }

    @Test
    void chainKeepsOnlyDeepestAsMaxBoundary() {
        DFSCode c1 = DFSCode.single(new EdgeTuple(0, 1, 1, 0, 1));
        DFSCode c2 = c1.extend(new EdgeTuple(1, 2, 1, 0, 2));
        DFSCode c3 = c2.extend(new EdgeTuple(2, 3, 2, 0, 3));

        BHUS.Sets sets = BHUS.extract(List.of(pr(c1), pr(c2), pr(c3)));
        assertEquals(Set.of(c3.key()), sets.maxBoundary());
        // parents of c2/c3 are present; c1's parent is the root (a HUS by convention)
        assertEquals(Set.of(), sets.minBoundary());
        assertEquals(Set.of(c3.key()), sets.all());
    }

    @Test
    void gapCreatesMinBoundary() {
        DFSCode c1 = DFSCode.single(new EdgeTuple(0, 1, 1, 0, 1));
        DFSCode c2 = c1.extend(new EdgeTuple(1, 2, 1, 0, 2));
        DFSCode c3 = c2.extend(new EdgeTuple(2, 3, 2, 0, 3));

        // c2 is NOT a HUS -> the path is broken (utility is not monotone)
        BHUS.Sets sets = BHUS.extract(List.of(pr(c1), pr(c3)));
        assertEquals(Set.of(c1.key(), c3.key()), sets.maxBoundary());
        assertEquals(Set.of(c3.key()), sets.minBoundary());
        assertEquals(Set.of(c1.key(), c3.key()), sets.all());
    }
}
