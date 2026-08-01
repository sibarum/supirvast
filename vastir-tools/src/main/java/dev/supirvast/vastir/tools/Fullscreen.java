package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;

import java.util.List;

/**
 * Screen-space shader primitives authored in the {@code core} IR — the building blocks for full-frame effects
 * (SDF ray-marchers, post-process passes, gradient fills) that need no vertex buffer at all.
 *
 * <p>The vertex stage draws a single oversized triangle that covers the whole viewport, with clip positions
 * derived purely from {@code gl_VertexIndex} — the classic {@code uv = ((idx << 1) & 2, idx & 2)},
 * {@code pos = uv * 2 - 1} trick. A caller issues {@code vkCmdDraw(cmd, 3, 1, 0, 0)} with an empty vertex-input
 * state; no attributes, no buffers. This is deliberately not a {@code GraphicsPipelineSpec} (that models an
 * interleaved vertex buffer) — a fullscreen pass has an <em>empty</em> vertex layout and is bound directly.
 *
 * <p>This lives in supirvast, not any front end: it is a general graphics primitive usable by any host that
 * owns a swapchain and render pass.
 */
public final class Fullscreen {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC2 = new Type.Vector(F32, 2);
    private static final Type.Vector VEC3 = new Type.Vector(F32, 3);
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    private Fullscreen() {
    }

    /** The conventional entry-point name both stages here use. */
    public static final String ENTRY_POINT = "main";

    /** The location the UV varying binds in both stages (vertex output, fragment input). */
    public static final int UV_LOCATION = 0;

    /** Byte size of {@link #STANDARD_UNIFORMS}: {@code vec2 resolution} (8) + {@code float time} (4). */
    public static final int STANDARD_UNIFORM_BYTES = 12;

    /**
     * The standard per-frame uniform block a fullscreen effect reads — ShaderToy-style {@code resolution} (the
     * framebuffer size in pixels) and {@code time} (seconds). The host pushes it every frame; a shader reads
     * {@code read(0)}/{@code read(1)} or ignores it. std430 offsets: resolution@0, time@8 (see
     * {@link #STANDARD_UNIFORM_BYTES}).
     */
    public static PushConstants standardUniforms() {
        return new PushConstants(List.of(
                new PushConstants.Member("resolution", VEC2),
                new PushConstants.Member("time", F32)));
    }

    /**
     * A vertex shader that emits a fullscreen triangle from {@code gl_VertexIndex} alone (three vertices, no
     * inputs). Writes only {@code gl_Position}; pair it with any fragment shader that shades by
     * {@code gl_FragCoord}-equivalent means — for now, a constant or a pass that will grow a UV varying.
     */
    public static byte[] triangleVertexSpirv() {
        Expr idx = new Expr.BuiltinRead(Builtin.VERTEX_INDEX);          // int32 in {0,1,2}
        Expr two = new Expr.ConstInt(Type.int32(), 2);
        Expr one = new Expr.ConstInt(Type.int32(), 1);
        // uv = ((idx << 1) & 2, idx & 2)  →  each component is 0 or 2
        Expr uxI = new Expr.Binary(BinaryOp.BIT_AND, new Expr.Binary(BinaryOp.SHIFT_LEFT, idx, one), two);
        Expr uyI = new Expr.Binary(BinaryOp.BIT_AND, idx, two);
        // pos = uv * 2 - 1  →  the triangle (-1,-1), (3,-1), (-1,3) covering the [-1,1] viewport
        Expr px = ndc(new Expr.Convert(uxI, F32));
        Expr py = ndc(new Expr.Convert(uyI, F32));
        Expr clip = new Expr.VectorConstruct(VEC4, List.of(px, py,
                new Expr.ConstFloat(F32, 0.0), new Expr.ConstFloat(F32, 1.0)));
        Region body = Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION, clip),
                new Statement.ReturnVoid());
        Function main = new Function(ENTRY_POINT, new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)))
                .toByteArray();
    }

    /**
     * A fullscreen-triangle vertex shader that also emits a {@code vUv} varying at {@link #UV_LOCATION} — the
     * screen-space coordinate spanning {@code [0,1]} across the visible viewport. Pair it with a fragment stage
     * that shades by {@code vUv} (a 2D field / SDF); the UV replaces the missing {@code gl_FragCoord} builtin.
     */
    public static byte[] triangleVertexWithUvSpirv() {
        Expr idx = new Expr.BuiltinRead(Builtin.VERTEX_INDEX);
        Expr two = new Expr.ConstInt(Type.int32(), 2);
        Expr one = new Expr.ConstInt(Type.int32(), 1);
        Expr ux = new Expr.Convert(new Expr.Binary(BinaryOp.BIT_AND,
                new Expr.Binary(BinaryOp.SHIFT_LEFT, idx, one), two), F32);   // 0 or 2
        Expr uy = new Expr.Convert(new Expr.Binary(BinaryOp.BIT_AND, idx, two), F32);
        Expr clip = new Expr.VectorConstruct(VEC4, List.of(ndc(ux), ndc(uy),
                new Expr.ConstFloat(F32, 0.0), new Expr.ConstFloat(F32, 1.0)));
        // vUv carries the raw {0,2} pair; across the visible viewport it interpolates to [0,1].
        InterfaceVar vUv = InterfaceVar.output("vUv", UV_LOCATION, VEC2);
        Expr uv = new Expr.VectorConstruct(VEC2, List.of(ux, uy));
        Region body = Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION, clip),
                new Statement.InterfaceWrite(vUv, uv),
                new Statement.ReturnVoid());
        Function main = new Function(ENTRY_POINT, new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)))
                .toByteArray();
    }

    /** {@code fragColor = vec4(r, g, b, a);} — the simplest fragment stage: a flat constant color. */
    public static byte[] constantColorFragmentSpirv(double r, double g, double b, double a) {
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Expr color = new Expr.VectorConstruct(VEC4, List.of(
                new Expr.ConstFloat(F32, r), new Expr.ConstFloat(F32, g),
                new Expr.ConstFloat(F32, b), new Expr.ConstFloat(F32, a)));
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor, color),
                new Statement.ReturnVoid());
        Function main = new Function(ENTRY_POINT, new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    /** {@code component * 2 - 1} — maps a {0,2} texel coordinate into a {-1,3} clip coordinate. */
    private static Expr ndc(Expr uvComponent) {
        Expr scaled = new Expr.Binary(BinaryOp.MUL, uvComponent, new Expr.ConstFloat(F32, 2.0));
        return new Expr.Binary(BinaryOp.SUB, scaled, new Expr.ConstFloat(F32, 1.0));
    }
}
