package edu.huflit.inchusm;

import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * {@link DFSCode#parse} must invert {@link DFSCode#key}: a pattern set mined in one run is
 * written as keys and read back to be matched against another graph, so any drift between
 * the two would silently match the wrong patterns.
 */
class DFSCodeParseTest {

    @Test
    void parseInvertsKeyOnAChain() {
        DFSCode code = DFSCode.of(List.of(
                new EdgeTuple(0, 1, 2, 0, 3),
                new EdgeTuple(1, 2, 3, 0, 1)));
        DFSCode back = DFSCode.parse(code.key());
        assertEquals(code.key(), back.key());
        assertEquals(code.nVertices(), back.nVertices());
        assertEquals(code.size(), back.size());
    }

    @Test
    void parseInvertsKeyOnACycleWithBackwardEdge() {
        DFSCode code = DFSCode.of(List.of(
                new EdgeTuple(0, 1, 2, 0, 2),
                new EdgeTuple(1, 2, 2, 0, 2),
                new EdgeTuple(2, 0, 2, 0, 2)));   // backward edge closes the triangle
        assertEquals(code.key(), DFSCode.parse(code.key()).key());
    }

    @Test
    void parseInvertsKeyOnASingleEdge() {
        DFSCode code = DFSCode.single(new EdgeTuple(0, 1, 1, 0, 1));
        assertEquals(code.key(), DFSCode.parse(code.key()).key());
    }

    @Test
    void malformedInputIsRejectedRatherThanSilentlyTruncated() {
        assertThrows(IllegalArgumentException.class, () -> DFSCode.parse(""));
        assertThrows(IllegalArgumentException.class, () -> DFSCode.parse("nonsense"));
        // trailing junk must not be ignored: it would mean matching a shorter pattern
        assertThrows(IllegalArgumentException.class, () -> DFSCode.parse("(0,1,1,0,1)xx"));
        // a gap between tuples must not be skipped over
        assertThrows(IllegalArgumentException.class, () -> DFSCode.parse("(0,1,1,0,1) (1,2,1,0,1)"));
    }
}
