package edu.huflit.inchusm.chain;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Independent reference for the chain miner, computed straight from the definitions.
 *
 * <p>Enumerates every sequence of transactions that walks through pairwise distinct accounts (an
 * open chain) or comes back to its first account (a closed chain), with no pruning of any kind,
 * checks the amount condition on the whole sequence, and collects the image set of every position.
 * It shares no code with the miner: not the graph, not the pattern class, not the predicate.
 */
final class ChainBruteForce {

    /** Which amount condition makes an occurrence valid. */
    enum Rule {
        /** Every pair of consecutive transactions at an intermediate account (the miner's rule). */
        PAIRWISE,
        /** Every transaction within theta of the first one (the cycle detectors' rule). */
        ANCHORED_FIRST
    }

    record Stat(int support, int[] imageCounts, List<Set<Integer>> images) {
    }

    private final int[] vlab;
    private final int[] src;
    private final int[] dst;
    private final int[] elab;
    private final double[] amt;
    private final double theta;
    private final int maxLength;
    private final int minCycle;
    private final boolean closedChains;
    private final Rule rule;
    private final Map<String, List<Set<Integer>>> images = new HashMap<>();

    ChainBruteForce(int[] vlab, int[] src, int[] dst, int[] elab, double[] amt, double theta,
                    int maxLength, int minCycle, boolean closedChains, Rule rule) {
        this.vlab = vlab;
        this.src = src;
        this.dst = dst;
        this.elab = elab;
        this.amt = amt;
        this.theta = theta;
        this.maxLength = maxLength;
        this.minCycle = minCycle;
        this.closedChains = closedChains;
        this.rule = rule;
    }

    /** Every pattern with at least one valid occurrence. */
    Map<String, Stat> all() {
        for (int e = 0; e < src.length; e++) {
            if (src[e] == dst[e]) continue;
            List<Integer> accounts = new ArrayList<>(List.of(src[e], dst[e]));
            List<Integer> txs = new ArrayList<>(List.of(e));
            walk(accounts, txs);
        }
        Map<String, Stat> out = new HashMap<>();
        for (Map.Entry<String, List<Set<Integer>>> kv : images.entrySet()) {
            int[] counts = new int[kv.getValue().size()];
            int min = Integer.MAX_VALUE;
            for (int i = 0; i < counts.length; i++) {
                counts[i] = kv.getValue().get(i).size();
                if (counts[i] < min) min = counts[i];
            }
            out.put(kv.getKey(), new Stat(min, counts, kv.getValue()));
        }
        return out;
    }

    /** Patterns whose support reaches minSupport. */
    Map<String, Stat> frequent(int minSupport) {
        Map<String, Stat> out = new HashMap<>();
        for (Map.Entry<String, Stat> kv : all().entrySet()) {
            if (kv.getValue().support() >= minSupport) out.put(kv.getKey(), kv.getValue());
        }
        return out;
    }

    private void walk(List<Integer> accounts, List<Integer> txs) {
        if (valid(txs)) record(false, accounts, txs);
        if (txs.size() == maxLength) return;
        int last = accounts.get(accounts.size() - 1);
        for (int f = 0; f < src.length; f++) {
            if (src[f] != last || dst[f] == last) continue;
            int next = dst[f];
            txs.add(f);
            if (next == accounts.get(0)) {
                if (closedChains && txs.size() >= minCycle && valid(txs)) record(true, accounts, txs);
            } else if (!accounts.contains(next)) {
                accounts.add(next);
                walk(accounts, txs);
                accounts.remove(accounts.size() - 1);
            }
            txs.remove(txs.size() - 1);
        }
    }

    /**
     * The amount condition on a whole sequence. For a closed chain the pair formed by the last and
     * the first transaction (at the origin) is not part of the sequence's consecutive pairs.
     */
    private boolean valid(List<Integer> txs) {
        if (rule == Rule.PAIRWISE) {
            for (int i = 0; i + 1 < txs.size(); i++) {
                double in = amt[txs.get(i)];
                double out = amt[txs.get(i + 1)];
                if (out < (1 - theta) * in || out > (1 + theta) * in) return false;
            }
        } else {
            double first = amt[txs.get(0)];
            for (int i = 1; i < txs.size(); i++) {
                double a = amt[txs.get(i)];
                if (a < (1 - theta) * first || a > (1 + theta) * first) return false;
            }
        }
        return true;
    }

    private void record(boolean closed, List<Integer> accounts, List<Integer> txs) {
        StringBuilder key = new StringBuilder(closed ? "C " : "O ").append(vlab[accounts.get(0)]);
        for (int i = 0; i < txs.size(); i++) {
            int to = closed && i == txs.size() - 1 ? accounts.get(0) : accounts.get(i + 1);
            key.append(" >").append(elab[txs.get(i)]).append(' ').append(vlab[to]);
        }
        int positions = closed ? txs.size() : txs.size() + 1;
        List<Set<Integer>> img = images.computeIfAbsent(key.toString(), k -> {
            List<Set<Integer>> l = new ArrayList<>();
            for (int i = 0; i < positions; i++) l.add(new TreeSet<>());
            return l;
        });
        for (int i = 0; i < positions; i++) img.get(i).add(accounts.get(i));
    }
}
