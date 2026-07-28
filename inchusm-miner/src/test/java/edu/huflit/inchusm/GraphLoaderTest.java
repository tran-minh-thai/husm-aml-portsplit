package edu.huflit.inchusm;

import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.graph.GraphLoader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GraphLoaderTest {

    @Test
    void parsesLgAndUtilities(@TempDir Path dir) throws IOException {
        Path lg = dir.resolve("g.lg");
        Files.writeString(lg, """
                t # 0
                # comment
                v 0 5
                v 1 6
                v 2 5
                e 0 1 2
                e 1 2
                """);
        Graph g = GraphLoader.loadLG(lg);
        assertEquals(3, g.n());
        assertEquals(2, g.m());
        assertEquals(5, g.vlabel(0));
        assertEquals(2, g.edgeLabelBetween(0, 1));
        assertEquals(0, g.edgeLabelBetween(1, 2)); // default edge label

        Path util = dir.resolve("u.tsv");
        Files.writeString(util, "5 1.5\n6 2.0\n");
        Map<Integer, Double> table = GraphLoader.loadUtilityTable(util);
        assertEquals(1.5, table.get(5));

        // gaussian utilities: deterministic per seed, clamped positive
        Map<Integer, Double> gauss = GraphLoader.gaussianUtilities(table.keySet(), 3.0, 1.0, 42L);
        assertEquals(gauss, GraphLoader.gaussianUtilities(table.keySet(), 3.0, 1.0, 42L));
        gauss.values().forEach(v -> org.junit.jupiter.api.Assertions.assertTrue(v > 0));
    }
}
