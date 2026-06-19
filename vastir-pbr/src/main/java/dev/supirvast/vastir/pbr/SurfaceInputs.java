package dev.supirvast.vastir.pbr;

import dev.supirvast.vastir.core.Expr;

/**
 * The fragment-stage values a {@link SurfaceFunction} may read when computing channels: the interpolated
 * geometric {@code worldNormal} and {@code worldPosition} (both {@code vec3}), supplied by the generated vertex
 * shader as location-bound varyings. A surface function builds its channel expressions from these plus
 * constants, math ({@link dev.supirvast.vastir.core.Expr.MathCall}), and storage-buffer reads.
 *
 * <p>(With no model transform yet, "world" space equals the model space the vertices are authored in.)
 */
public record SurfaceInputs(Expr worldNormal, Expr worldPosition) {
}
