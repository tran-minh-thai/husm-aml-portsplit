package edu.huflit.inchusm.chain;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Executable checks of the statements the miner relies on, computed by the reference enumerator
 * (no miner code involved), each paired with a rule that must fail the same check so the check is
 * known to be able to fail.
 */
class ChainTheoryTest {

    private record Violations(int comparisons, int prefix, int suffix) {
    }

    /** Text key of the open chain made of positions first..first+edges of a pattern key. */
    private static String openSub(String key, int first, int edges) {
        String[] tok = key.split(" ");
        // tok[0] is O or C, then label, then pairs (">y", "x")
        StringBuilder sb = new StringBuilder("O ").append(tok[1 + 2 * first]);
        for (int i = 0; i < edges; i++) {
            sb.append(' ').append(tok[2 + 2 * (first + i)]).append(' ').append(tok[3 + 2 * (first + i)]);
        }
        return sb.toString();
    }

    private static int length(String key) {
        return (key.split(" ").length - 2) / 2;
    }

    /** Support of every chain against that of its prefix and its suffix (the two chains one shorter). */
    private static Violations check(Map<String, ChainBruteForce.Stat> all) {
        int comparisons = 0;
        int prefix = 0;
        int suffix = 0;
        for (Map.Entry<String, ChainBruteForce.Stat> kv : all.entrySet()) {
            String key = kv.getKey();
            int m = length(key);
            if (m < 2) continue;
            int s = kv.getValue().support();
            ChainBruteForce.Stat pre = all.get(openSub(key, 0, m - 1));
            ChainBruteForce.Stat suf = all.get(openSub(key, 1, m - 1));
            comparisons++;
            if (s > (pre == null ? 0 : pre.support())) prefix++;
            if (s > (suf == null ? 0 : suf.support())) suffix++;
        }
        return new Violations(comparisons, prefix, suffix);
    }

    private static Violations sweep(ChainBruteForce.Rule rule, boolean closed) {
        Random rnd = new Random(20261011L);
        int comparisons = 0;
        int prefix = 0;
        int suffix = 0;
        for (int it = 0; it < 2000; it++) {
            ChainMinerTest.Instance x = ChainMinerTest.randomInstance(rnd);
            double theta = ChainMinerTest.THETAS[1 + it % 3];
            Map<String, ChainBruteForce.Stat> all = new ChainBruteForce(x.vlab(), x.src(), x.dst(), x.elab(),
                    x.amt(), theta, 4, 2, closed, rule).all();
            if (!closed) all.keySet().removeIf(k -> k.startsWith("C"));
            Violations v = check(all);
            comparisons += v.comparisons();
            prefix += v.prefix();
            suffix += v.suffix();
        }
        System.out.printf("%s closed=%s: %d comparisons, %d prefix violations, %d suffix violations%n",
                rule, closed, comparisons, prefix, suffix);
        return new Violations(comparisons, prefix, suffix);
    }

    @Test
    void pairwiseConservationGivesSupportBelowPrefixAndSuffix() {
        Violations v = sweep(ChainBruteForce.Rule.PAIRWISE, true);
        assertTrue(v.comparisons() > 3000, "comparisons: " + v.comparisons());
        assertEquals(0, v.prefix());
        assertEquals(0, v.suffix());
    }

    @Test
    void anchoringOnTheFirstTransactionKeepsPrefixesButNotSuffixes() {
        Violations v = sweep(ChainBruteForce.Rule.ANCHORED_FIRST, false);
        assertTrue(v.comparisons() > 3000, "comparisons: " + v.comparisons());
        assertEquals(0, v.prefix());
        assertTrue(v.suffix() > 0, "the check must be able to fail: suffix violations " + v.suffix());
    }

    @Test
    void constrainingTotalsAtEachAccountIsNotInherited() {
        // a -> v (100), b -> v (100), v -> c (200): money in at v equals money out, so the
        // two-in-one-out occurrence conserves; its sub-occurrence a -> v -> c (100 then 200) does not.
        double theta = 0.2;
        double in = 100 + 100;
        double out = 200;
        assertTrue(out >= (1 - theta) * in && out <= (1 + theta) * in, "the larger occurrence conserves totals");
        assertFalse(200 >= (1 - theta) * 100 && 200 <= (1 + theta) * 100, "its chain sub-occurrence does not");
        // In a chain every intermediate account has one transaction in and one out, so its total
        // condition is the pairwise condition, which the reference shows to be inherited above.
    }
}
