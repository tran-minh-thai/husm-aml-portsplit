package edu.huflit.inchusm.chain;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import org.junit.jupiter.api.Test;

class ChainMinerTest {

    // ------------------------------------------------------------------ helpers

    /** Transactions given as {source, target, label, amount}. */
    static TransactionGraph graph(int[] vlab, double[][] tx) {
        int m = tx.length;
        int[] s = new int[m];
        int[] t = new int[m];
        int[] l = new int[m];
        double[] a = new double[m];
        for (int i = 0; i < m; i++) {
            s[i] = (int) tx[i][0];
            t[i] = (int) tx[i][1];
            l[i] = (int) tx[i][2];
            a[i] = tx[i][3];
        }
        return new TransactionGraph(vlab, s, t, l, a);
    }

    static Map<String, ChainMiner.Result> mine(TransactionGraph g, double theta, int minSupport, int maxLength,
                                               int minCycle) {
        Map<String, ChainMiner.Result> out = new HashMap<>();
        for (ChainMiner.Result r : new ChainMiner(g, new ChainMiner.Params(theta, minSupport, maxLength, minCycle,
                true)).mine()) {
            out.put(r.pattern().key(), r);
        }
        return out;
    }

    // ------------------------------------------------------------------ cases computed by hand

    @Test
    void conservationBoundsAreInclusive() {
        // theta = 0.25 and amounts that are exact in binary: 100 -> [75, 125]
        double[][] upper = {{0, 1, 0, 100}, {1, 2, 0, 125}};
        double[][] lower = {{0, 1, 0, 100}, {1, 2, 0, 75}};
        double[][] above = {{0, 1, 0, 100}, {1, 2, 0, Math.nextUp(125.0)}};
        double[][] below = {{0, 1, 0, 100}, {1, 2, 0, Math.nextDown(75.0)}};
        int[] v = {0, 0, 0};
        assertTrue(mine(graph(v, upper), 0.25, 1, 2, 3).containsKey("O 0 >0 0 >0 0"));
        assertTrue(mine(graph(v, lower), 0.25, 1, 2, 3).containsKey("O 0 >0 0 >0 0"));
        assertFalse(mine(graph(v, above), 0.25, 1, 2, 3).containsKey("O 0 >0 0 >0 0"));
        assertFalse(mine(graph(v, below), 0.25, 1, 2, 3).containsKey("O 0 >0 0 >0 0"));
    }

    @Test
    void thetaZeroDemandsEqualAmounts() {
        int[] v = {0, 0, 0};
        assertTrue(mine(graph(v, new double[][] {{0, 1, 0, 7}, {1, 2, 0, 7}}), 0.0, 1, 2, 3)
                .containsKey("O 0 >0 0 >0 0"));
        assertFalse(mine(graph(v, new double[][] {{0, 1, 0, 7}, {1, 2, 0, Math.nextUp(7.0)}}), 0.0, 1, 2, 3)
                .containsKey("O 0 >0 0 >0 0"));
    }

    @Test
    void cycleWithFeesIsFoundOnlyFromItsOrigin() {
        // a -> b -> c -> a keeping 90% at each hop (the way one synthetic generator builds cycles).
        // Intermediate pairs: 100 -> 90 and 90 -> 81, both within 0.2. From origin b the pair at a
        // would be 81 -> 100 (ratio 1.2346), outside 0.2; from origin c likewise.
        int[] v = {0, 0, 0};
        double[][] tx = {{0, 1, 0, 100}, {1, 2, 0, 90}, {2, 0, 0, 81}};
        Map<String, ChainMiner.Result> r = mine(graph(v, tx), 0.2, 1, 3, 3);
        ChainMiner.Result cycle = r.get("C 0 >0 0 >0 0 >0 0");
        assertEquals(1, cycle.support());
        assertArrayEquals(new int[] {1, 1, 1}, cycle.imageCounts());
        // open chains of length 2: a-b-c and b-c-a are valid, c-a-b is not
        ChainMiner.Result two = r.get("O 0 >0 0 >0 0");
        assertArrayEquals(new int[] {2, 2, 2}, two.imageCounts());
        assertEquals(2, two.support());
        // the open chain of length 3 would need a fourth distinct account
        assertFalse(r.containsKey("O 0 >0 0 >0 0 >0 0"));
    }

    @Test
    void parallelTransactionsCountWhenOneOfThemConserves() {
        int[] v = {0, 0, 0};
        double[][] tx = {{0, 1, 0, 10}, {0, 1, 0, 100}, {1, 2, 0, 95}};
        assertEquals(1, mine(graph(v, tx), 0.1, 1, 2, 3).get("O 0 >0 0 >0 0").support());
    }

    @Test
    void accountsOfAnOpenChainAreDistinct() {
        // a=0, b=1, c=2; a->b, b->a, a->c all 100
        int[] v = {0, 0, 0};
        double[][] tx = {{0, 1, 0, 100}, {1, 0, 0, 100}, {0, 2, 0, 100}};
        Map<String, ChainMiner.Result> r = mine(graph(v, tx), 0.0, 1, 3, 2);
        // only b -> a -> c is an open chain of length 2
        assertArrayEquals(new int[] {1, 1, 1}, r.get("O 0 >0 0 >0 0").imageCounts());
        // a -> b -> a and b -> a -> b are closed chains of length 2 (origins a and b)
        assertArrayEquals(new int[] {2, 2}, r.get("C 0 >0 0 >0 0").imageCounts());
        assertFalse(r.containsKey("O 0 >0 0 >0 0 >0 0"));
    }

    @Test
    void labelsSeparatePatternsAndSupportIsTheSmallestImage() {
        // two sources with label 1 send to one hub with label 0, which forwards to a sink with label 2
        int[] v = {1, 1, 0, 2};
        double[][] tx = {{0, 2, 5, 100}, {1, 2, 5, 100}, {2, 3, 6, 100}};
        Map<String, ChainMiner.Result> r = mine(graph(v, tx), 0.0, 1, 2, 3);
        ChainMiner.Result p = r.get("O 1 >5 0 >6 2");
        assertArrayEquals(new int[] {2, 1, 1}, p.imageCounts());
        assertEquals(1, p.support());
        // at minimum support 2 nothing qualifies: the hub and the sink are single accounts
        assertTrue(mine(graph(v, tx), 0.0, 2, 2, 3).isEmpty());
    }

    @Test
    void selfTransactionsAreDroppedAndCounted() {
        TransactionGraph g = graph(new int[] {0, 0}, new double[][] {{0, 0, 0, 5}, {0, 1, 0, 5}});
        assertEquals(1, g.droppedSelfTransactions());
        assertEquals(1, g.numTransactions());
    }

    @Test
    void nonPositiveAmountsAreRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> graph(new int[] {0, 0}, new double[][] {{0, 1, 0, 0}}));
        assertThrows(IllegalArgumentException.class,
                () -> graph(new int[] {0, 0}, new double[][] {{0, 1, 0, Double.NaN}}));
    }

    // ------------------------------------------------------------------ against the reference

    static final double[] AMOUNTS = {48, 60, 64, 75, 80, 90, 96, 100, 120, 125, 128, 150};
    static final double[] THETAS = {0.0, 0.2, 0.25, 0.5, Double.POSITIVE_INFINITY};

    record Instance(int[] vlab, int[] src, int[] dst, int[] elab, double[] amt) {
        TransactionGraph graph() {
            return new TransactionGraph(vlab, src, dst, elab, amt);
        }
    }

    /** Small random graphs rich in parallel transactions, cycles, self-transactions and exact boundaries. */
    static Instance randomInstance(Random rnd) {
        int n = 3 + rnd.nextInt(5);
        int vLabels = 1 + rnd.nextInt(2);
        int eLabels = 1 + rnd.nextInt(2);
        int m = 4 + rnd.nextInt(15);
        int[] vlab = new int[n];
        for (int i = 0; i < n; i++) vlab[i] = rnd.nextInt(vLabels);
        int[] s = new int[m];
        int[] t = new int[m];
        int[] l = new int[m];
        double[] a = new double[m];
        for (int i = 0; i < m; i++) {
            s[i] = rnd.nextInt(n);
            t[i] = rnd.nextInt(10) == 0 ? s[i] : rnd.nextInt(n);
            l[i] = rnd.nextInt(eLabels);
            a[i] = rnd.nextInt(4) == 0 ? 40 + 120 * rnd.nextDouble() : AMOUNTS[rnd.nextInt(AMOUNTS.length)];
        }
        return new Instance(vlab, s, t, l, a);
    }

    @Test
    void minerEqualsReferenceOnRandomGraphs() {
        Random rnd = new Random(20261010L);
        int graphs = 0;
        int compared = 0;
        int closedCompared = 0;
        int longCompared = 0;
        int boundaryRuns = 0;
        for (int it = 0; it < 3000; it++) {
            Instance x = randomInstance(rnd);
            double theta = THETAS[it % THETAS.length];
            int minSupport = 1 + rnd.nextInt(3);
            int maxLength = 3 + rnd.nextInt(2);
            int minCycle = 2 + rnd.nextInt(2);
            Map<String, ChainBruteForce.Stat> want = new ChainBruteForce(x.vlab(), x.src(), x.dst(), x.elab(),
                    x.amt(), theta, maxLength, minCycle, true, ChainBruteForce.Rule.PAIRWISE).frequent(minSupport);
            boolean[] marked = new boolean[x.vlab().length];
            for (int v = 0; v < marked.length; v++) marked[v] = rnd.nextBoolean();
            ChainMiner miner = new ChainMiner(x.graph(), new ChainMiner.Params(theta, minSupport, maxLength, minCycle,
                    true)).marked(marked).keepImages(true);
            Map<String, ChainMiner.Result> got = new HashMap<>();
            for (ChainMiner.Result r : miner.mine()) got.put(r.pattern().key(), r);
            Map<String, TreeSet<Integer>> roles = new HashMap<>();
            assertEquals(new TreeSet<>(want.keySet()), new TreeSet<>(got.keySet()), "pattern set, instance " + it);
            for (Map.Entry<String, ChainBruteForce.Stat> kv : want.entrySet()) {
                ChainMiner.Result r = got.get(kv.getKey());
                assertEquals(kv.getValue().support(), r.support(), kv.getKey() + " support, instance " + it);
                assertArrayEquals(kv.getValue().imageCounts(), r.imageCounts(), kv.getKey() + " images, instance " + it);
                int[] wantMarked = new int[r.images().length];
                for (int i = 0; i < wantMarked.length; i++) {
                    assertEquals(kv.getValue().images().get(i), toSet(r.images()[i]), kv.getKey() + " image set " + i);
                    for (int v : kv.getValue().images().get(i)) if (marked[v]) wantMarked[i]++;
                }
                assertArrayEquals(wantMarked, r.markedCounts(), kv.getKey() + " marked counts, instance " + it);
                String kind = r.pattern().closed() ? "closed " : "open ";
                int m = r.pattern().length();
                if (m >= 2) {
                    TreeSet<Integer> mid = roles.computeIfAbsent(kind + m + " intermediate", k -> new TreeSet<>());
                    for (int i = 1; i < m; i++) mid.addAll(kv.getValue().images().get(i));
                }
                if (r.pattern().closed()) {
                    roles.computeIfAbsent(kind + m + " origin", k -> new TreeSet<>()).addAll(kv.getValue().images().get(0));
                }
                compared++;
                if (r.pattern().closed()) closedCompared++;
                if (r.pattern().length() >= 3) longCompared++;
            }
            Map<String, TreeSet<Integer>> gotRoles = new HashMap<>();
            miner.accountsByRole().forEach((k, b) -> gotRoles.put(k, toSet(b.stream().toArray())));
            assertEquals(roles, gotRoles, "accounts by role, instance " + it);
            graphs++;
            if (theta == 0.25) boundaryRuns++;
        }
        // the comparison must not be vacuous
        System.out.printf("reference comparison: %d graphs, %d patterns compared, %d closed, %d of length >= 3,"
                + " %d runs at theta 0.25%n", graphs, compared, closedCompared, longCompared, boundaryRuns);
        assertTrue(compared > 2000, "patterns compared: " + compared);
        assertTrue(closedCompared > 100, "closed chains compared: " + closedCompared);
        assertTrue(longCompared > 300, "chains of length >= 3 compared: " + longCompared);
    }

    static TreeSet<Integer> toSet(int[] a) {
        TreeSet<Integer> s = new TreeSet<>();
        for (int v : a) s.add(v);
        return s;
    }

    @Test
    void markingNeverChangesWhatIsMined() {
        Random rnd = new Random(11);
        for (int it = 0; it < 200; it++) {
            Instance x = randomInstance(rnd);
            ChainMiner.Params p = new ChainMiner.Params(0.25, 1 + rnd.nextInt(2), 4, 2, true);
            boolean[] marked = new boolean[x.vlab().length];
            for (int v = 0; v < marked.length; v++) marked[v] = rnd.nextBoolean();
            List<ChainMiner.Result> a = new ChainMiner(x.graph(), p).mine();
            List<ChainMiner.Result> b = new ChainMiner(x.graph(), p).marked(marked).mine();
            assertEquals(a.size(), b.size());
            for (int i = 0; i < a.size(); i++) {
                assertEquals(a.get(i).pattern().key(), b.get(i).pattern().key());
                assertArrayEquals(a.get(i).imageCounts(), b.get(i).imageCounts());
            }
        }
    }

    @Test
    void loadingAFileGivesTheSameChainsAsTheArrays(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir)
            throws java.io.IOException {
        Random rnd = new Random(5);
        for (int it = 0; it < 100; it++) {
            Instance x = randomInstance(rnd);
            // ids far from 0..n-1 and accounts declared after the transactions
            StringBuilder sb = new StringBuilder("# test graph\n");
            for (int e = 0; e < x.src().length; e++) {
                sb.append("e ").append(1000 + 7L * x.src()[e]).append(' ').append(1000 + 7L * x.dst()[e]).append(' ')
                        .append(x.elab()[e]).append(' ').append(x.amt()[e]).append('\n');
            }
            for (int v = x.vlab().length - 1; v >= 0; v--) {
                sb.append("v ").append(1000 + 7L * v).append(' ').append(x.vlab()[v]).append('\n');
            }
            java.nio.file.Path f = dir.resolve("g" + it + ".tg");
            java.nio.file.Files.writeString(f, sb.toString());
            TransactionGraph loaded = TransactionGraph.load(f);
            ChainMiner.Params p = new ChainMiner.Params(0.2, 1, 4, 2, true);
            List<ChainMiner.Result> a = new ChainMiner(x.graph(), p).keepImages(true).mine();
            List<ChainMiner.Result> b = new ChainMiner(loaded, p).keepImages(true).mine();
            assertEquals(a.size(), b.size());
            for (int i = 0; i < a.size(); i++) {
                assertEquals(a.get(i).pattern().key(), b.get(i).pattern().key());
                for (int q = 0; q < a.get(i).images().length; q++) {
                    TreeSet<Long> want = new TreeSet<>();
                    for (int v : a.get(i).images()[q]) want.add(1000 + 7L * v);
                    TreeSet<Long> have = new TreeSet<>();
                    for (int v : b.get(i).images()[q]) have.add(loaded.originalId(v));
                    assertEquals(want, have);
                }
            }
        }
    }

    @Test
    void minerResultsAreDeterministic() {
        Random rnd = new Random(7);
        for (int it = 0; it < 50; it++) {
            Instance x = randomInstance(rnd);
            ChainMiner.Params p = new ChainMiner.Params(0.25, 1, 4, 2, true);
            List<ChainMiner.Result> a = new ChainMiner(x.graph(), p).mine();
            List<ChainMiner.Result> b = new ChainMiner(x.graph(), p).mine();
            assertEquals(a.size(), b.size());
            for (int i = 0; i < a.size(); i++) {
                assertEquals(a.get(i).pattern().key(), b.get(i).pattern().key());
                assertArrayEquals(a.get(i).imageCounts(), b.get(i).imageCounts());
            }
        }
    }
}
