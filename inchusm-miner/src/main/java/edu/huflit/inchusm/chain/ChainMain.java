package edu.huflit.inchusm.chain;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Command line for the conserved flow chain miner.
 *
 * <pre>
 * java -cp inchusm-miner.jar edu.huflit.inchusm.chain.ChainMain --graph G.tg --theta 0.2 \
 *      --min-support 10 --max-length 6 [--min-cycle 3] [--no-closed] --out patterns.tsv
 * </pre>
 *
 * The output table has one row per chain whose support reaches the threshold: kind (open or
 * closed), length, position labels, transaction labels, support and the size of every image set.
 * Work counters are printed to standard output; they do not depend on the machine. The one line
 * that does (elapsed time) is marked as such so recording tools can tell it apart.
 */
public final class ChainMain {

    private ChainMain() {
    }

    public static void main(String[] args) throws IOException {
        Map<String, String> opt = parse(args);
        Path graphPath = Path.of(required(opt, "--graph"));
        Path out = Path.of(required(opt, "--out"));
        ChainMiner.Params params = new ChainMiner.Params(
                Double.parseDouble(required(opt, "--theta")),
                Integer.parseInt(required(opt, "--min-support")),
                Integer.parseInt(required(opt, "--max-length")),
                Integer.parseInt(opt.getOrDefault("--min-cycle", "3")),
                !opt.containsKey("--no-closed"));

        TransactionGraph g = TransactionGraph.load(graphPath);
        System.out.printf("graph %s: %d accounts, %d transactions (%d self-transactions dropped)%n",
                graphPath.getFileName(), g.numAccounts(), g.numTransactions(), g.droppedSelfTransactions());
        System.out.printf("params: theta=%s min_support=%d max_length=%d min_cycle=%d closed_chains=%s%n",
                params.theta(), params.minSupport(), params.maxLength(), params.minCycleLength(),
                params.closedChains());

        ChainMiner miner = new ChainMiner(g, params);
        long t0 = System.nanoTime();
        List<ChainMiner.Result> results = miner.mine();
        long elapsedMs = (System.nanoTime() - t0) / 1_000_000;

        write(out, results);
        ChainMiner.Counters c = miner.counters();
        System.out.printf("candidates=%d pruned_by_domain=%d searches=%d found=%d failed=%d early_stops=%d%n",
                c.candidates, c.prunedByDomain, c.searches, c.searchesFound, c.searchesFailed, c.earlyStops);
        for (Map.Entry<Integer, long[]> e : c.frequentByLength.entrySet()) {
            System.out.printf("length %d: %d open, %d closed%n", e.getKey(), e.getValue()[0], e.getValue()[1]);
        }
        System.out.printf("patterns written: %d -> %s%n", results.size(), out.getFileName());
        System.out.printf("mining elapsed ms: %d (machine-dependent)%n", elapsedMs);
    }

    static void write(Path out, List<ChainMiner.Result> results) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("kind\tlength\taccount_labels\ttransaction_labels\tsupport\timage_counts\n");
            for (ChainMiner.Result r : results) {
                ChainPattern p = r.pattern();
                w.write((p.closed() ? "closed" : "open") + "\t" + p.length() + "\t" + join(p.vlab()) + "\t"
                        + join(p.elab()) + "\t" + r.support() + "\t" + join(r.imageCounts()) + "\n");
            }
        }
    }

    private static String join(int[] a) {
        return Arrays.stream(a).mapToObj(Integer::toString).collect(Collectors.joining(","));
    }

    private static Map<String, String> parse(String[] args) {
        Map<String, String> opt = new java.util.HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (!a.startsWith("--")) throw new IllegalArgumentException("unexpected argument '" + a + "'");
            if (a.equals("--no-closed")) {
                opt.put(a, "");
            } else {
                if (i + 1 >= args.length) throw new IllegalArgumentException(a + " needs a value");
                opt.put(a, args[++i]);
            }
        }
        return opt;
    }

    private static String required(Map<String, String> opt, String key) {
        String v = opt.get(key);
        if (v == null) throw new IllegalArgumentException("missing " + key);
        return v;
    }
}
