package edu.huflit.inchusm.chain;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.Test;

/** The anchored rule of the enumerator against the reference, and its accounts by role. */
class ChainEnumeratorAnchoredTest {

    @Test
    void anchoredEnumeratorEqualsReference() {
        Random rnd = new Random(20261017L);
        int compared = 0;
        int differFromPairwise = 0;
        for (int it = 0; it < 2000; it++) {
            ChainMinerTest.Instance x = ChainMinerTest.randomInstance(rnd);
            double theta = ChainMinerTest.THETAS[it % 4];                 // finite thetas only
            int minSupport = 1 + rnd.nextInt(3);
            int maxLength = 3 + rnd.nextInt(2);
            ChainMiner.Params p = new ChainMiner.Params(theta, minSupport, maxLength, 2, false);
            Map<String, ChainBruteForce.Stat> want = new ChainBruteForce(x.vlab(), x.src(), x.dst(), x.elab(), x.amt(),
                    theta, maxLength, 2, false, ChainBruteForce.Rule.ANCHORED_FIRST).frequent(minSupport);
            ChainEnumerator en = new ChainEnumerator(x.graph(), p).anchored(true);
            Map<String, ChainMiner.Result> got = new HashMap<>();
            for (ChainMiner.Result r : en.mine()) got.put(r.pattern().key(), r);
            assertEquals(new TreeSet<>(want.keySet()), new TreeSet<>(got.keySet()), "instance " + it);
            Map<String, TreeSet<Integer>> roles = new HashMap<>();
            for (Map.Entry<String, ChainBruteForce.Stat> kv : want.entrySet()) {
                assertArrayEquals(kv.getValue().imageCounts(), got.get(kv.getKey()).imageCounts(), kv.getKey());
                int m = got.get(kv.getKey()).pattern().length();
                if (m >= 2) {
                    TreeSet<Integer> mid = roles.computeIfAbsent("open " + m + " intermediate", k -> new TreeSet<>());
                    for (int i = 1; i < m; i++) mid.addAll(kv.getValue().images().get(i));
                }
                compared++;
            }
            Map<String, TreeSet<Integer>> gotRoles = new HashMap<>();
            en.accountsByRole().forEach((k, b) -> gotRoles.put(k, ChainMinerTest.toSet(b.stream().toArray())));
            assertEquals(roles, gotRoles, "accounts by role, instance " + it);
            Set<String> pairwise = new ChainBruteForce(x.vlab(), x.src(), x.dst(), x.elab(), x.amt(), theta, maxLength, 2,
                    false, ChainBruteForce.Rule.PAIRWISE).frequent(minSupport).keySet();
            if (!pairwise.equals(want.keySet())) differFromPairwise++;
        }
        System.out.printf("anchored enumerator against reference: %d patterns compared; %d instances where the"
                + " anchored and pairwise rules report different chains%n", compared, differFromPairwise);
        assertTrue(compared > 3000, "compared " + compared);
        assertTrue(differFromPairwise > 50, "the two rules must differ somewhere: " + differFromPairwise);
    }

    @Test
    void anchoredRuleRefusesClosedChains() {
        ChainMinerTest.Instance x = ChainMinerTest.randomInstance(new Random(1));
        assertThrows(IllegalArgumentException.class,
                () -> new ChainEnumerator(x.graph(), new ChainMiner.Params(0.2, 1, 3, 3, true)).anchored(true));
    }
}
