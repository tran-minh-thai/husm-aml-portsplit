package edu.huflit.inchusm.chain;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import org.junit.jupiter.api.Test;

/** The enumeration baseline must give the reference's output before any time is measured on it. */
class ChainEnumeratorTest {

    @Test
    void enumeratorEqualsReferenceOnRandomGraphs() {
        Random rnd = new Random(20261013L);
        int compared = 0;
        int closed = 0;
        for (int it = 0; it < 2000; it++) {
            ChainMinerTest.Instance x = ChainMinerTest.randomInstance(rnd);
            double theta = ChainMinerTest.THETAS[it % ChainMinerTest.THETAS.length];
            int minSupport = 1 + rnd.nextInt(3);
            int maxLength = 3 + rnd.nextInt(2);
            int minCycle = 2 + rnd.nextInt(2);
            boolean withClosed = it % 4 != 0;
            Map<String, ChainBruteForce.Stat> want = new ChainBruteForce(x.vlab(), x.src(), x.dst(), x.elab(),
                    x.amt(), theta, maxLength, minCycle, withClosed, ChainBruteForce.Rule.PAIRWISE).frequent(minSupport);
            Map<String, ChainMiner.Result> got = new HashMap<>();
            for (ChainMiner.Result r : new ChainEnumerator(x.graph(),
                    new ChainMiner.Params(theta, minSupport, maxLength, minCycle, withClosed)).mine()) {
                got.put(r.pattern().key(), r);
            }
            assertEquals(new TreeSet<>(want.keySet()), new TreeSet<>(got.keySet()), "instance " + it);
            for (Map.Entry<String, ChainBruteForce.Stat> kv : want.entrySet()) {
                assertArrayEquals(kv.getValue().imageCounts(), got.get(kv.getKey()).imageCounts(), kv.getKey());
                compared++;
                if (kv.getKey().startsWith("C")) closed++;
            }
        }
        System.out.printf("enumerator against reference: %d patterns compared, %d closed%n", compared, closed);
        assertTrue(compared > 3000 && closed > 300, "compared " + compared + ", closed " + closed);
    }
}
