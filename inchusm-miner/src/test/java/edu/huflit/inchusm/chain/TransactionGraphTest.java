package edu.huflit.inchusm.chain;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import java.util.TreeSet;
import org.junit.jupiter.api.Test;

/** Range queries on accounts with many transactions, checked against a linear scan. */
class TransactionGraphTest {

    @Test
    void conservingRangeEqualsLinearScanOnHubs() {
        Random rnd = new Random(31);
        int queries = 0;
        int nonEmpty = 0;
        for (int it = 0; it < 30; it++) {
            int n = 6;
            int m = 400 + rnd.nextInt(400);                   // hubs with hundreds of transactions each
            int[] vlab = new int[n];
            int[] s = new int[m];
            int[] t = new int[m];
            int[] l = new int[m];
            double[] a = new double[m];
            for (int e = 0; e < m; e++) {
                s[e] = rnd.nextInt(n);
                t[e] = (s[e] + 1 + rnd.nextInt(n - 1)) % n;
                l[e] = rnd.nextInt(3);
                a[e] = rnd.nextInt(3) == 0 ? 50 + rnd.nextInt(100) : 50 + 100 * rnd.nextDouble();
            }
            TransactionGraph g = new TransactionGraph(vlab, s, t, l, a);
            for (int q = 0; q < 200; q++) {
                int v = rnd.nextInt(n);
                int lab = rnd.nextInt(3);
                double x = 50 + 100 * rnd.nextDouble();
                double theta = new double[] {0.0, 0.1, 0.2, 0.5}[rnd.nextInt(4)];
                double lo = (1 - theta) * x;
                double hi = (1 + theta) * x;
                // outgoing: amounts b with lo <= b <= hi
                TreeSet<Integer> want = new TreeSet<>();
                for (int e = 0; e < g.numTransactions(); e++) {
                    if (g.source(e) == v && g.label(e) == lab && g.amount(e) >= lo && g.amount(e) <= hi) want.add(e);
                }
                int[] r = g.narrow(true, g.outByLabel(v, lab), b -> b >= lo, b -> b <= hi);
                TreeSet<Integer> have = new TreeSet<>();
                for (int i = r[0]; i < r[1]; i++) have.add(g.outSlot(i));
                assertEquals(want, have);
                // incoming: amounts c with (1 + theta) c >= x and (1 - theta) c <= x
                TreeSet<Integer> wantIn = new TreeSet<>();
                for (int e = 0; e < g.numTransactions(); e++) {
                    double c = g.amount(e);
                    if (g.target(e) == v && g.label(e) == lab && (1 + theta) * c >= x && (1 - theta) * c <= x) {
                        wantIn.add(e);
                    }
                }
                int[] ri = g.narrow(false, g.inByLabel(v, lab), c -> (1 + theta) * c >= x, c -> (1 - theta) * c <= x);
                TreeSet<Integer> haveIn = new TreeSet<>();
                for (int i = ri[0]; i < ri[1]; i++) haveIn.add(g.inSlot(i));
                assertEquals(wantIn, haveIn);
                queries++;
                if (!want.isEmpty() && !wantIn.isEmpty()) nonEmpty++;
            }
        }
        System.out.printf("range queries: %d, with both directions non-empty: %d%n", queries, nonEmpty);
        assertTrue(nonEmpty > 1000, "non-empty queries: " + nonEmpty);
    }
}
