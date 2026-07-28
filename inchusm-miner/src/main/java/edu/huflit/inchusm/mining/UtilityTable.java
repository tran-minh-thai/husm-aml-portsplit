package edu.huflit.inchusm.mining;

import java.util.LinkedHashMap;
import java.util.Map;

/** External utilities of vertex labels (HUSM Fig. 2b). Missing labels are an error. */
public final class UtilityTable {

    private final Map<Integer, Double> u;

    public UtilityTable(Map<Integer, Double> u) {
        this.u = new LinkedHashMap<>(u);
    }

    public double of(int label) {
        Double d = u.get(label);
        if (d == null) {
            throw new IllegalArgumentException("no external utility for vertex label " + label);
        }
        return d;
    }

    public Map<Integer, Double> asMap() {
        return Map.copyOf(u);
    }
}
