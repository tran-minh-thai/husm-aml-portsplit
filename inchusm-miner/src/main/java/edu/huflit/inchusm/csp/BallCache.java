package edu.huflit.inchusm.csp;

import edu.huflit.inchusm.graph.Graph;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Cache of BFS balls of a FIXED radius (= MER) on the data graph. Shared across
 * all pattern evaluations of one mining run: the same centers recur constantly
 * (every f(I0) of every pattern), and the balls double as the ESV of Def. 11.
 */
public final class BallCache {

    private final Graph g;
    private final int radius;
    private final Map<Integer, int[]> cache = new HashMap<>();

    public BallCache(Graph g, int radius) {
        this.g = g;
        this.radius = radius;
    }

    public int radius() {
        return radius;
    }

    /** Sorted members of the ball around center (center included). */
    public int[] ball(int center) {
        return cache.computeIfAbsent(center, c -> g.ball(c, radius));
    }

    public boolean inBall(int center, int x) {
        return Arrays.binarySearch(ball(center), x) >= 0;
    }
}
