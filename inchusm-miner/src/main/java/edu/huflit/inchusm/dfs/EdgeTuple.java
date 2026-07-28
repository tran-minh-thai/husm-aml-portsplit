package edu.huflit.inchusm.dfs;

/**
 * One DFS-code edge: the 5-tuple (i, j, l_i, l_e, l_j) of gSpan (Yan &amp; Han 2002,
 * Def. 1). Forward edge iff i &lt; j (tree edge, j is a new vertex); backward iff
 * i &gt; j (closes a cycle back to a rightmost-path vertex).
 */
public record EdgeTuple(int i, int j, int li, int le, int lj) {

    public boolean isForward() {
        return i < j;
    }

    @Override
    public String toString() {
        return "(" + i + "," + j + "," + li + "," + le + "," + lj + ")";
    }
}
