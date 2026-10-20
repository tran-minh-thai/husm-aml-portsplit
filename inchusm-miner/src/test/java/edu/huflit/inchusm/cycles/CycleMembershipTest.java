package edu.huflit.inchusm.cycles;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicLongArray;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CycleMembershipTest {

    @TempDir
    Path tmp;

    /** From the definition: accounts at any position of a sequence of m distinct accounts joined by transactions into a cycle. */
    static TreeSet<Long> bruteForce(int n, List<int[]> edges, int m) {
        boolean[][] adj = new boolean[n][n];
        for (int[] e : edges) if (e[0] != e[1]) adj[e[0]][e[1]] = true;
        TreeSet<Long> on = new TreeSet<>();
        int[] seq = new int[m];
        for (int s = 0; s < n; s++) {
            seq[0] = s;
            extend(adj, seq, 1, m, on);
        }
        return on;
    }

    private static void extend(boolean[][] adj, int[] seq, int k, int m, TreeSet<Long> on) {
        if (k == m) {
            if (adj[seq[m - 1]][seq[0]]) for (int a : seq) on.add(100L + a);
            return;
        }
        for (int w = 0; w < adj.length; w++) {
            if (!adj[seq[k - 1]][w]) continue;
            boolean used = false;
            for (int i = 0; i < k; i++) used |= seq[i] == w;
            if (used) continue;
            seq[k] = w;
            extend(adj, seq, k + 1, m, on);
        }
    }

    private Path write(int n, List<int[]> edges, String name) throws IOException {
        StringBuilder sb = new StringBuilder("# test graph\n");
        for (int v = 0; v < n; v++) sb.append("v ").append(100 + v).append(" 0\n");     // ids are not indices
        for (int[] e : edges) sb.append("e ").append(100 + e[0]).append(' ').append(100 + e[1]).append(" 0 ").append(e[2]).append('\n');
        Path p = tmp.resolve(name + ".tg");
        Files.writeString(p, sb.toString());
        return p;
    }

    private static TreeSet<Long> computed(CycleMembership g, int m, int threads) {
        AtomicLongArray b = g.members(m, threads);
        TreeSet<Long> on = new TreeSet<>();
        for (int v = 0; v < g.n; v++) if (CycleMembership.isSet(b, v)) on.add(g.originalId[v]);
        return on;
    }

    private static List<int[]> randomGraph(Random r, int n, double density) {
        List<int[]> edges = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (r.nextDouble() >= density) continue;
                int copies = 1 + (r.nextDouble() < 0.2 ? 1 : 0);                // parallel transactions
                for (int c = 0; c < copies; c++) edges.add(new int[] {u, v, 1 + r.nextInt(100)});   // u == v: self-transaction
            }
        }
        return edges;
    }

    @Test
    void agreesWithTheDefinitionOnRandomGraphs() throws IOException {
        Random r = new Random(20261020);
        int compared = 0, nonEmpty = 0, graphs = 0;
        for (int i = 0; i < 1500; i++) {
            int n = 3 + r.nextInt(6);
            double density = i % 3 == 0 ? 0.6 + 0.35 * r.nextDouble() : 0.1 + 0.4 * r.nextDouble();   // a third dense
            List<int[]> edges = randomGraph(r, n, density);
            Path p = write(n, edges, "g" + i);
            graphs++;
            for (String orientation : new String[] {"auto", "forward", "reverse"}) {
                CycleMembership g = CycleMembership.load(p, orientation);
                for (int m = 3; m <= 6; m++) {
                    TreeSet<Long> expect = bruteForce(n, edges, m);
                    for (int threads : new int[] {1, 3}) {
                        assertEquals(expect, computed(g, m, threads), "graph " + i + " m " + m + " " + orientation + " threads " + threads);
                        compared++;
                    }
                    if (!expect.isEmpty() && orientation.equals("auto")) nonEmpty++;
                }
            }
        }
        // not vacuous: many comparisons have accounts on cycles, at every length
        assertTrue(nonEmpty > 1500, "comparisons with a non-empty cycle set: " + nonEmpty);
        assertEquals(graphs * 3 * 4 * 2, compared);
    }

    /** Every single search, without the witnesses other searches record: finds a cycle exactly when v is on one. */
    private static void eachSearch(CycleMembership g, int n, List<int[]> edges, int m, String where) {
        TreeSet<Long> expect = bruteForce(n, edges, m);
        for (int v = 0; v < g.n; v++) {
            if (g.outOff[v + 1] == g.outOff[v]) continue;
            int[] c = CycleMembership.Scratch.get(g.n).search(g, v, m);
            if (c != null) g.check(c, m);
            assertEquals(expect.contains(g.originalId[v]), c != null, where + " account " + g.originalId[v] + " m " + m);
        }
    }

    @Test
    void everySearchAloneAgreesWithTheDefinition() throws IOException {
        Random r = new Random(7);
        int positive = 0;
        for (int i = 0; i < 600; i++) {
            int n = 4 + r.nextInt(5);
            List<int[]> edges = randomGraph(r, n, 0.3 + 0.6 * r.nextDouble());
            Path p = write(n, edges, "s" + i);
            for (String orientation : new String[] {"forward", "reverse"}) {
                CycleMembership g = CycleMembership.load(p, orientation);
                for (int m = 3; m <= 6; m++) {
                    eachSearch(g, n, edges, m, "graph " + i + " " + orientation);
                    positive += bruteForce(n, edges, m).size();
                }
            }
        }
        assertTrue(positive > 3000, "accounts on cycles over the searches: " + positive);
        // v = 0 -> a -> b -> z -> c -> 0 is the only cycle of length 5 through 0 that the forward search can close, and
        // the tabulated accounts before 0 reached from z are a, b and c in this order: the first two are the leaving path
        int v = 0, a = 1, b = 2, c = 3, z = 4;
        List<int[]> edges = List.of(new int[] {v, a, 1}, new int[] {a, b, 1}, new int[] {b, z, 1}, new int[] {z, a, 1},
                new int[] {z, b, 1}, new int[] {z, c, 1}, new int[] {a, v, 1}, new int[] {b, v, 1}, new int[] {c, v, 1});
        CycleMembership g = CycleMembership.load(write(5, edges, "third"), "forward");
        assertTrue(bruteForce(5, edges, 5).contains(100L + v));
        eachSearch(g, 5, edges, 5, "third alternative");
    }

    @Test
    void blockedSecondToLastAccountsNeedEveryTabulatedAlternative() throws IOException {
        // complete digraphs: every candidate y of the table is blocked for some leaving path, at lengths 4 and 5
        for (int n = 4; n <= 7; n++) {
            List<int[]> edges = new ArrayList<>();
            for (int u = 0; u < n; u++) for (int v = 0; v < n; v++) if (u != v) edges.add(new int[] {u, v, 5});
            CycleMembership g = CycleMembership.load(write(n, edges, "k" + n), "auto");
            for (int m = 3; m <= Math.min(n, 6); m++) assertEquals(bruteForce(n, edges, m), computed(g, m, 2), "K" + n + " m " + m);
        }
        // a 5-cycle whose only closing pair goes through accounts adjacent to the leaving path
        List<int[]> edges = new ArrayList<>(List.of(new int[] {0, 1, 1}, new int[] {1, 2, 1}, new int[] {2, 3, 1},
                new int[] {3, 4, 1}, new int[] {4, 0, 1}, new int[] {3, 1, 1}, new int[] {3, 2, 1}, new int[] {1, 0, 1}, new int[] {2, 0, 1}));
        CycleMembership g = CycleMembership.load(write(5, edges, "blocked"), "forward");
        assertEquals(bruteForce(5, edges, 5), computed(g, 5, 1));
        assertEquals(5, computed(g, 5, 1).size());
    }

    @Test
    void edgesOutsideStronglyConnectedComponentsAreRemoved() throws IOException {
        // two 3-cycles joined by one bridge, and a tail; the bridge and the tail lie on no cycle
        List<int[]> edges = List.of(new int[] {0, 1, 1}, new int[] {1, 2, 1}, new int[] {2, 0, 1}, new int[] {2, 3, 1},
                new int[] {3, 4, 1}, new int[] {4, 5, 1}, new int[] {5, 3, 1}, new int[] {5, 6, 1});
        CycleMembership g = CycleMembership.load(write(7, edges, "bridge"), "auto");
        assertEquals(6, g.sccAccounts);
        assertEquals(6, g.sccEdges);
        assertEquals(bruteForce(7, edges, 3), computed(g, 3, 1));
        assertEquals(6, computed(g, 3, 1).size());
        assertTrue(computed(g, 4, 1).isEmpty());
    }

    @Test
    void refusesMoreThanOneLabel() throws IOException {
        Path a = tmp.resolve("labels.tg");
        Files.writeString(a, "v 1 0\nv 2 1\ne 1 2 0 5\n");
        assertThrows(IllegalArgumentException.class, () -> CycleMembership.load(a, "auto"));
        Path b = tmp.resolve("elabels.tg");
        Files.writeString(b, "v 1 0\nv 2 0\ne 1 2 0 5\ne 2 1 3 5\n");
        assertThrows(IllegalArgumentException.class, () -> CycleMembership.load(b, "auto"));
    }

    @Test
    void writesTheAccountsOfReportedClosedChains() throws Exception {
        List<int[]> edges = List.of(new int[] {0, 1, 1}, new int[] {1, 2, 1}, new int[] {2, 0, 1}, new int[] {2, 0, 7},
                new int[] {3, 4, 1}, new int[] {4, 5, 1}, new int[] {5, 6, 1}, new int[] {6, 3, 1}, new int[] {3, 3, 2});
        Path g = write(7, edges, "out");
        Path out = tmp.resolve("accounts.tsv");
        CycleMembership.main(new String[] {"--graph", g.toString(), "--min-support", "4", "--out", out.toString(), "--threads", "2"});
        List<String> lines = Files.readAllLines(out);
        assertEquals("kind\tlength\trole\taccount", lines.get(0));
        // length 3 has support 3 < 4: not reported; length 4 has support 4: origin and intermediate rows
        assertEquals(List.of("closed\t4\torigin\t103", "closed\t4\torigin\t104", "closed\t4\torigin\t105", "closed\t4\torigin\t106",
                "closed\t4\tintermediate\t103", "closed\t4\tintermediate\t104", "closed\t4\tintermediate\t105",
                "closed\t4\tintermediate\t106"), lines.subList(1, lines.size()));
    }
}
