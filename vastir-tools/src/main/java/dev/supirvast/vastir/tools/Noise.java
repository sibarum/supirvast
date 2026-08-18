package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.type.Type;

import java.util.List;

/**
 * Procedural noise authored in the {@code core} IR — pure {@link Expr} trees that lower to <em>both</em> GPU
 * SPIR-V and the CPU (Truffle) backend, so a field built from this noise evaluates identically on the GPU it is
 * drawn on and the CPU that queries it (collision, line-of-sight, physics). Each method emits fresh IR per call
 * and references only the standard GLSL-450 math ({@code sin}, {@code floor}, {@code fract}, {@code mix},
 * {@code smoothstep}, {@code dot}) that every backend implements.
 *
 * <p>The workhorse is {@link #value2}: classic value noise — a grid of hashed pseudo-random values, smoothly
 * (smoothstep-)interpolated, i.e. a "Gaussian-ish filtered random grid." {@link #fbm2} sums octaves of it for
 * multi-scale detail (fractional Brownian motion). Both return values in roughly {@code [0, 1]}.
 *
 * <p>These are pure expressions (no local variables), so a sub-term like {@code floor(p)} is re-emitted at each
 * use rather than bound once; the duplication is small (a handful of ops per octave) and lets the noise drop into
 * any expression context — including an inlined SDF — without needing its own {@code core} function.
 *
 * <p>Like {@link Fullscreen}, this lives in supirvast as a general graphics primitive, usable by any host.
 */
public final class Noise {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC2 = new Type.Vector(F32, 2);

    private Noise() {
    }

    /**
     * A 2D hash → pseudo-random scalar in {@code [0, 1)}: {@code fract(sin(dot(p, k)) * m)} with the classic
     * large-constant vector {@code k} and multiplier {@code m}. Deterministic in {@code p}; not statistically
     * rigorous, but cheap and standard for procedural terrain/detail.
     *
     * @param p a {@code vec2} expression (typically an integer grid coordinate)
     */
    public static Expr hash2(Expr p) {
        Expr k = vec2(127.1, 311.7);
        return Expr.MathCall.fract(mul(Expr.MathCall.sin(Expr.MathCall.dot(p, k)), f(43758.5453)));
    }

    /**
     * Value noise at {@code p}: hash the four surrounding integer grid corners and bilinearly interpolate with
     * smoothstep weights, giving a smooth (C1) scalar field in roughly {@code [0, 1]}. One world unit spans one
     * noise cell — scale {@code p} before calling to set the feature size.
     *
     * @param p a {@code vec2} sample position
     */
    public static Expr value2(Expr p) {
        Expr i = Expr.MathCall.floor(p);                 // cell origin
        Expr fp = Expr.MathCall.fract(p);                // position within the cell
        Expr u = quinticFade(fp);                        // C2 blend weights (see quinticFade)

        Expr a = hash2(i);
        Expr b = hash2(add(i, vec2(1.0, 0.0)));
        Expr c = hash2(add(i, vec2(0.0, 1.0)));
        Expr d = hash2(add(i, vec2(1.0, 1.0)));

        Expr ux = new Expr.VectorExtract(u, 0);
        Expr uy = new Expr.VectorExtract(u, 1);
        Expr bottom = Expr.MathCall.mix(a, b, ux);
        Expr top = Expr.MathCall.mix(c, d, ux);
        return Expr.MathCall.mix(bottom, top, uy);
    }

    /**
     * Fractional Brownian motion: {@code octaves} of {@link #value2} summed with lacunarity 2 (each octave twice
     * the frequency) and gain 0.5 (each octave half the amplitude). Adds fine detail on top of coarse structure —
     * "lots of curvature at various render distances." Result is roughly in {@code [0, 1]}. Scale {@code p} before
     * calling to set the base feature size.
     *
     * @param p       a {@code vec2} sample position
     * @param octaves how many octaves to sum (1 = plain {@link #value2}); must be positive
     */
    public static Expr fbm2(Expr p, int octaves) {
        if (octaves < 1) {
            throw new IllegalArgumentException("octaves must be >= 1, got " + octaves);
        }
        Expr sum = f(0.0);
        double amplitude = 0.5;
        double frequency = 1.0;
        double angle = 0.0;
        for (int o = 0; o < octaves; o++) {
            // Rotate each octave by an irrational-ish angle so the axis-aligned value-noise grid does not line up
            // across octaves and telegraph through as blocky ridges.
            Expr sample = value2(scale(rotate(p, angle), frequency));
            sum = add(sum, mul(sample, f(amplitude)));
            frequency *= 2.0;
            amplitude *= 0.5;
            angle += 0.617;
        }
        return sum;
    }

    /**
     * Gradient (Perlin) noise at {@code p} — signed, roughly {@code [-0.7, 0.7]}, zero at the integer grid points.
     * Unlike {@link #value2} (which interpolates random <em>values</em> and always shows square grid patches),
     * this interpolates the dot of a random unit <em>gradient</em> at each corner with the offset to {@code p},
     * with a quintic fade. Because the field passes through zero at every grid point and its slope there is set by
     * a random direction, the result looks organic — no quilted blocks. This is the go-to for smooth terrain.
     *
     * @param p a {@code vec2} sample position
     */
    public static Expr perlin2(Expr p) {
        Expr i = Expr.MathCall.floor(p);
        Expr fp = Expr.MathCall.fract(p);
        Expr u = quinticFade(fp);

        Expr n00 = gradDot(i, p);
        Expr n10 = gradDot(add(i, vec2(1.0, 0.0)), p);
        Expr n01 = gradDot(add(i, vec2(0.0, 1.0)), p);
        Expr n11 = gradDot(add(i, vec2(1.0, 1.0)), p);

        Expr ux = new Expr.VectorExtract(u, 0);
        Expr uy = new Expr.VectorExtract(u, 1);
        Expr nx0 = Expr.MathCall.mix(n00, n10, ux);
        Expr nx1 = Expr.MathCall.mix(n01, n11, ux);
        return Expr.MathCall.mix(nx0, nx1, uy);
    }

    /**
     * Fractional Brownian motion over {@link #perlin2}: {@code octaves} summed with lacunarity 2 / gain 0.5, each
     * octave rotated (including the first) so no octave's grid aligns with the axes. Returns a <em>signed</em>
     * field centred near 0 (roughly {@code [-0.6, 0.6]}) — use it directly as a height, no {@code -0.5} centring.
     *
     * @param p       a {@code vec2} sample position
     * @param octaves how many octaves to sum (>= 1)
     */
    public static Expr fbmPerlin2(Expr p, int octaves) {
        if (octaves < 1) {
            throw new IllegalArgumentException("octaves must be >= 1, got " + octaves);
        }
        Expr sum = f(0.0);
        double amplitude = 0.5;
        double frequency = 1.0;
        double angle = 0.3;   // start rotated so even the dominant first octave is off-axis
        for (int o = 0; o < octaves; o++) {
            sum = add(sum, mul(perlin2(scale(rotate(p, angle), frequency)), f(amplitude)));
            frequency *= 2.0;
            amplitude *= 0.5;
            angle += 0.617;
        }
        return sum;
    }

    /** Dot of a random unit gradient at grid corner {@code corner} with the offset {@code p - corner}. The gradient
     *  direction is a hashed angle, so it is deterministic per corner. Uses cos/sin (both backends implement them). */
    private static Expr gradDot(Expr corner, Expr p) {
        Expr angle = mul(hash2(corner), f(6.2831853));                 // hashed direction in [0, 2pi)
        Expr g = new Expr.VectorConstruct(VEC2, List.of(Expr.MathCall.cos(angle), Expr.MathCall.sin(angle)));
        return Expr.MathCall.dot(g, sub(p, corner));
    }

    // --- tiny IR-authoring helpers (mirrors the style of the core authoring code) ---

    private static Expr f(double v) {
        return new Expr.ConstFloat(F32, v);
    }

    private static Expr vec2(double a, double b) {
        return new Expr.VectorConstruct(VEC2, List.of(f(a), f(b)));
    }

    private static Expr add(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }

    private static Expr sub(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.SUB, a, b);
    }

    private static Expr mul(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.MUL, a, b);
    }

    /** {@code vec2 * scalar} via broadcast (vector*scalar is not a core primitive). */
    private static Expr scale(Expr vec, double s) {
        return new Expr.Binary(BinaryOp.MUL, vec, vec2(s, s));
    }

    /**
     * Perlin's quintic fade, per component: {@code 6t^5 - 15t^4 + 10t^3}, i.e. {@code t^3 (t (6t - 15) + 10)}.
     * Unlike cubic {@code smoothstep} (C1), the quintic is C2 — its first derivative is zero AND smooth at cell
     * edges, so interpolated value noise has a continuous gradient and does not show faceted creases along the
     * grid when shaded by its finite-difference normal. Built from add/sub/mul only (no new backend op).
     */
    private static Expr quinticFade(Expr t) {
        Expr a = sub(mul(t, vec2(6.0, 6.0)), vec2(15.0, 15.0));   // 6t - 15
        Expr b = add(mul(t, a), vec2(10.0, 10.0));                // t(6t - 15) + 10
        Expr t3 = mul(mul(t, t), t);                             // t^3
        return mul(t3, b);
    }

    /**
     * Rotate a {@code vec2} by a compile-time constant {@code angle} (radians). The sin/cos are folded in Java, so
     * this emits only mul/sub/add on the components — used to decorrelate fBm octaves from the noise grid.
     */
    private static Expr rotate(Expr p, double angle) {
        double c = Math.cos(angle);
        double s = Math.sin(angle);
        Expr px = new Expr.VectorExtract(p, 0);
        Expr py = new Expr.VectorExtract(p, 1);
        Expr rx = sub(mul(px, f(c)), mul(py, f(s)));
        Expr ry = add(mul(px, f(s)), mul(py, f(c)));
        return new Expr.VectorConstruct(VEC2, List.of(rx, ry));
    }
}
