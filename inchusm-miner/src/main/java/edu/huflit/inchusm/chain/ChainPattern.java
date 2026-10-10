package edu.huflit.inchusm.chain;

import java.util.Arrays;

/**
 * A flow-chain pattern of length m (m transactions).
 *
 * <p>Open chain: accounts at positions 0..m with labels {@code vlab[0..m]}, transaction i (1-based)
 * goes from position i-1 to position i with label {@code elab[i-1]}; the m+1 accounts of an
 * occurrence are pairwise distinct.
 *
 * <p>Closed chain (m >= 2): the same, except that position m is position 0 again: money leaves the
 * origin (position 0) and comes back to it. The m accounts at positions 0..m-1 are pairwise
 * distinct, and {@code vlab[m] == vlab[0]}. The origin is part of the pattern: two rotations of a
 * cycle are two patterns.
 *
 * <p>Conservation is required at every intermediate position 1..m-1 (between the transaction that
 * arrives there and the one that leaves). The origin of a closed chain is not intermediate: the
 * transaction that returns is not a forwarding of the one that left, so their pair is not
 * constrained.
 */
public record ChainPattern(int[] vlab, int[] elab, boolean closed) {

    public ChainPattern {
        if (elab.length < 1 || vlab.length != elab.length + 1) {
            throw new IllegalArgumentException("a chain of m transactions has m+1 position labels");
        }
        if (closed && (elab.length < 2 || vlab[0] != vlab[elab.length])) {
            throw new IllegalArgumentException("a closed chain needs m >= 2 and the same label at both ends");
        }
        vlab = vlab.clone();
        elab = elab.clone();
    }

    /** Number of transactions. */
    public int length() {
        return elab.length;
    }

    /** Number of distinct account positions: m+1 for an open chain, m for a closed one. */
    public int positions() {
        return closed ? elab.length : elab.length + 1;
    }

    /** Canonical text form, e.g. {@code O 3 >0 5 >1 3} or {@code C 3 >0 5 >1 3}. */
    public String key() {
        StringBuilder sb = new StringBuilder(closed ? "C " : "O ");
        sb.append(vlab[0]);
        for (int i = 0; i < elab.length; i++) {
            sb.append(" >").append(elab[i]).append(' ').append(vlab[i + 1]);
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof ChainPattern p && closed == p.closed
                && Arrays.equals(vlab, p.vlab) && Arrays.equals(elab, p.elab);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * Arrays.hashCode(vlab) + Arrays.hashCode(elab)) + (closed ? 1 : 0);
    }

    @Override
    public String toString() {
        return key();
    }
}
