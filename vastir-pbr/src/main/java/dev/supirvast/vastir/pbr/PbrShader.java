package dev.supirvast.vastir.pbr;

import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Expr.MathCall;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.tools.GraphicsPipelineSpec;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static dev.supirvast.vastir.pbr.Shade.VEC2;
import static dev.supirvast.vastir.pbr.Shade.VEC3;
import static dev.supirvast.vastir.pbr.Shade.VEC4;
import static dev.supirvast.vastir.pbr.Shade.add;
import static dev.supirvast.vastir.pbr.Shade.div;
import static dev.supirvast.vastir.pbr.Shade.divScalar;
import static dev.supirvast.vastir.pbr.Shade.f;
import static dev.supirvast.vastir.pbr.Shade.mul;
import static dev.supirvast.vastir.pbr.Shade.scale;
import static dev.supirvast.vastir.pbr.Shade.splat3;
import static dev.supirvast.vastir.pbr.Shade.sub;
import static dev.supirvast.vastir.pbr.Shade.vec3;
import static dev.supirvast.vastir.pbr.Shade.vec4;

/**
 * Generates a lit vertex+fragment shader pair from a declared set of {@link Channel}s and a
 * {@link SurfaceFunction}, the ShaderLab-style split: you describe the <em>surface</em> (what each channel is),
 * and this fills in the <em>lighting</em> (a Cook-Torrance metallic-roughness BRDF). The generated shaders are
 * {@code core} IR lowered to SPIR-V and packaged as a {@link GraphicsPipelineSpec} the previewer can render.
 *
 * <p>The vertex shader follows the previewer's v1 contract (position @0, normal @1) and passes the geometric
 * normal and position through as varyings; the fragment shader evaluates the surface function, then shades it
 * with one hardcoded directional light plus a small ambient term, Reinhard tone-maps, and gamma-encodes.
 * Lighting is in the space the geometry is authored in (no model/view/projection transform yet). Channels not
 * supplied by the surface function take sensible defaults.
 */
public final class PbrShader {

    // Hardcoded scene lighting (a single directional light) — a later pass can promote these to uniforms.
    private static final double PI = 3.141592653589793;
    private static final double DIELECTRIC_F0 = 0.04;
    private static final double EPSILON = 1.0e-4;
    private static final double INV_GAMMA = 1.0 / 2.2;

    private final Set<Channel> channels;
    private final SurfaceFunction surface;
    private final Integer environmentBinding;   // null = flat ambient; else IBL from a cubemap at this binding

    private PbrShader(Set<Channel> channels, SurfaceFunction surface, Integer environmentBinding) {
        this.channels = channels;
        this.surface = surface;
        this.environmentBinding = environmentBinding;
    }

    /**
     * A material exposing {@code channels}, whose values are computed by {@code surface}. The surface function
     * must return exactly the declared channels, each with an expression matching {@link Channel#type()};
     * channels left out of {@code channels} use defaults. Lit by one directional light plus a small flat ambient.
     */
    public static PbrShader create(Set<Channel> channels, SurfaceFunction surface) {
        return new PbrShader(normalize(channels), surface, null);
    }

    /**
     * Like {@link #create}, but the ambient term is image-based: a cubemap bound at descriptor set 0,
     * {@code environmentBinding} supplies the diffuse irradiance (sampled along the normal) and the specular
     * reflection (sampled along the reflected view), so metals reflect the environment instead of going dark.
     */
    public static PbrShader createWithEnvironment(Set<Channel> channels, SurfaceFunction surface,
            int environmentBinding) {
        return new PbrShader(normalize(channels), surface, environmentBinding);
    }

    private static Set<Channel> normalize(Set<Channel> channels) {
        return channels.isEmpty() ? EnumSet.noneOf(Channel.class) : EnumSet.copyOf(channels);
    }

    /** The generated vertex+fragment pair as a renderable spec (each stage lowered to SPIR-V). */
    public GraphicsPipelineSpec spec() {
        return GraphicsPipelineSpec.standard(
                lower(vertexFunction(), ShaderStage.VERTEX),
                lower(fragmentFunction(), ShaderStage.FRAGMENT));
    }

    private static byte[] lower(Function function, ShaderStage stage) {
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(function, stage)))
                .toByteArray();
    }

    /** {@code gl_Position = vec4(position, 1); pass the normal, position, and uv through as varyings.} */
    public Function vertexFunction() {
        InterfaceVar position = InterfaceVar.input("position", 0, VEC3);
        InterfaceVar normal = InterfaceVar.input("normal", 1, VEC3);
        InterfaceVar uv = InterfaceVar.input("uv", 2, VEC2);
        InterfaceVar worldNormal = InterfaceVar.output("vWorldNormal", 0, VEC3);
        InterfaceVar worldPosition = InterfaceVar.output("vWorldPosition", 1, VEC3);
        InterfaceVar varyingUv = InterfaceVar.output("vUv", 2, VEC2);

        Region body = Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION,
                        vec4(new Expr.InterfaceRead(position), f(1))),
                new Statement.InterfaceWrite(worldNormal, new Expr.InterfaceRead(normal)),
                new Statement.InterfaceWrite(worldPosition, new Expr.InterfaceRead(position)),
                new Statement.InterfaceWrite(varyingUv, new Expr.InterfaceRead(uv)),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    /** Evaluates the surface function, then shades it with the Cook-Torrance lighting model. */
    public Function fragmentFunction() {
        InterfaceVar worldNormal = InterfaceVar.input("vWorldNormal", 0, VEC3);
        InterfaceVar worldPosition = InterfaceVar.input("vWorldPosition", 1, VEC3);
        InterfaceVar uv = InterfaceVar.input("vUv", 2, VEC2);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);

        SurfaceInputs inputs = new SurfaceInputs(new Expr.InterfaceRead(worldNormal),
                new Expr.InterfaceRead(worldPosition), new Expr.InterfaceRead(uv));
        Map<Channel, Expr> provided = surface.evaluate(inputs);
        validate(provided);

        Body b = new Body();
        // Bind each channel once (so the lighting model reuses values instead of re-evaluating the surface).
        Expr albedo = b.let("albedo", resolved(provided, Channel.ALBEDO, inputs));
        Expr metallic = b.let("metallic", resolved(provided, Channel.METALLIC, inputs));
        Expr roughness = b.let("roughness", resolved(provided, Channel.ROUGHNESS, inputs));
        Expr ao = b.let("ao", resolved(provided, Channel.AO, inputs));
        Expr emissive = b.let("emissive", resolved(provided, Channel.EMISSIVE, inputs));
        Expr opacity = b.let("opacity", resolved(provided, Channel.OPACITY, inputs));
        Expr shadingNormal = resolved(provided, Channel.NORMAL, inputs);

        Expr color = cookTorrance(b, albedo, metallic, roughness, ao, emissive, shadingNormal, inputs);

        b.statements.add(new Statement.InterfaceWrite(fragColor, vec4(color, opacity)));
        b.statements.add(new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), new Region(b.statements));
    }

    /** The metallic-roughness microfacet BRDF: GGX distribution, Smith geometry, Schlick Fresnel. */
    private Expr cookTorrance(Body b, Expr albedo, Expr metallic, Expr roughness, Expr ao, Expr emissive,
            Expr shadingNormal, SurfaceInputs inputs) {
        Expr n = b.let("N", MathCall.normalize(shadingNormal));
        // The previewer clears depth to 1 with LESS and no projection, so the nearer (smaller-z) hemisphere is
        // visible — i.e. the camera sits on the -Z side. Place the view and light there to light the front.
        Expr v = b.let("V", MathCall.normalize(sub(vec3(0, 0, -1.0), inputs.worldPosition())));
        Expr l = b.let("L", MathCall.normalize(vec3(0.35, 0.45, -0.9)));   // up-right, mostly toward the camera
        Expr h = b.let("H", MathCall.normalize(add(v, l)));

        Expr nDotL = b.let("NdotL", MathCall.max(MathCall.dot(n, l), f(0)));
        Expr nDotV = b.let("NdotV", MathCall.max(MathCall.dot(n, v), f(EPSILON)));
        Expr nDotH = b.let("NdotH", MathCall.max(MathCall.dot(n, h), f(0)));
        Expr vDotH = b.let("VdotH", MathCall.max(MathCall.dot(v, h), f(0)));

        Expr alpha = b.let("alpha", mul(roughness, roughness));
        Expr alpha2 = b.let("alpha2", mul(alpha, alpha));

        // Normal distribution (GGX/Trowbridge-Reitz).
        Expr dDenom = b.let("Dden", add(mul(mul(nDotH, nDotH), sub(alpha2, f(1))), f(1)));
        Expr distribution = b.let("D", div(alpha2, mul(f(PI), mul(dDenom, dDenom))));

        // Geometry (Smith with Schlick-GGX, direct-lighting k).
        Expr rPlus1 = b.let("rPlus1", add(roughness, f(1)));
        Expr k = b.let("k", div(mul(rPlus1, rPlus1), f(8)));
        Expr gv = b.let("Gv", div(nDotV, add(mul(nDotV, sub(f(1), k)), k)));
        Expr gl = b.let("Gl", div(nDotL, add(mul(nDotL, sub(f(1), k)), k)));
        Expr geometry = b.let("G", mul(gv, gl));

        // Fresnel (Schlick); F0 lerps dielectric 0.04 toward albedo by metalness.
        Expr f0 = b.let("F0", MathCall.mix(splat3(f(DIELECTRIC_F0)), albedo, splat3(metallic)));
        Expr fresnelPow = b.let("Fpow", MathCall.pow(sub(f(1), vDotH), f(5)));
        Expr fresnel = b.let("F", add(f0, scale(sub(splat3(f(1)), f0), fresnelPow)));

        // Specular = D*G*F / (4*NdotV*NdotL).
        Expr dg = b.let("DG", mul(distribution, geometry));
        Expr specDenom = b.let("specDen", add(mul(f(4), mul(nDotV, nDotL)), f(EPSILON)));
        Expr specular = b.let("specular", divScalar(scale(fresnel, dg), specDenom));

        // Diffuse = (1-F)*(1-metallic)*albedo / PI (energy conservation: no diffuse from metals).
        Expr kd = b.let("kd", scale(sub(splat3(f(1)), fresnel), sub(f(1), metallic)));
        Expr diffuse = b.let("diffuse", divScalar(mul(kd, albedo), f(PI)));

        // Outgoing radiance from the single light, plus ambient (flat or image-based) and emissive.
        Expr lightColor = vec3(4, 4, 4);
        Expr lo = b.let("Lo", scale(mul(add(diffuse, specular), lightColor), nDotL));
        Expr ambient = ambientTerm(b, albedo, metallic, ao, f0, n, v, nDotV);
        Expr color = b.let("color", add(add(ambient, lo), emissive));

        // Reinhard tone-map then gamma-encode for display.
        Expr mapped = b.let("mapped", div(color, add(color, splat3(f(1)))));
        return b.let("gamma", MathCall.pow(mapped, splat3(f(INV_GAMMA))));
    }

    /**
     * The ambient contribution: a small flat term by default, or — when an environment is set — image-based
     * lighting that samples the cubemap for diffuse irradiance (along {@code n}) and the specular reflection
     * (along the reflected view), so metals reflect the environment.
     */
    private Expr ambientTerm(Body b, Expr albedo, Expr metallic, Expr ao, Expr f0, Expr n, Expr v, Expr nDotV) {
        if (environmentBinding == null) {
            return b.let("ambient", scale(scale(albedo, f(0.05)), ao));
        }
        Expr reflected = b.let("R", MathCall.reflect(new Expr.Unary(UnaryOp.NEGATE, v), n));
        Expr envSpecular = b.let("envSpec", Shade.xyz(Shade.sampleCube("environment", environmentBinding, reflected)));
        Expr envDiffuse = b.let("envDiff", Shade.xyz(Shade.sampleCube("environment", environmentBinding, n)));
        Expr fresnel = b.let("Fa", add(f0, scale(sub(splat3(f(1)), f0), MathCall.pow(sub(f(1), nDotV), f(5)))));
        Expr kd = b.let("kdA", scale(sub(splat3(f(1)), fresnel), sub(f(1), metallic)));
        Expr diffuse = b.let("ambDiff", mul(mul(kd, envDiffuse), albedo));
        Expr specular = b.let("ambSpec", mul(fresnel, envSpecular));
        return b.let("ambient", scale(add(diffuse, specular), ao));
    }

    private Expr resolved(Map<Channel, Expr> provided, Channel channel, SurfaceInputs inputs) {
        Expr value = provided.get(channel);
        if (value != null) {
            return value;
        }
        return switch (channel) {
            case ALBEDO -> vec3(0.8, 0.8, 0.8);
            case NORMAL -> inputs.worldNormal();
            case METALLIC -> f(0);
            case ROUGHNESS -> f(0.5);
            case AO -> f(1);
            case EMISSIVE -> vec3(0, 0, 0);
            case OPACITY -> f(1);
        };
    }

    private void validate(Map<Channel, Expr> provided) {
        if (!provided.keySet().equals(channels)) {
            throw new IllegalArgumentException("surface function must set exactly the declared channels "
                    + channels + ", but set " + provided.keySet());
        }
        provided.forEach((channel, expr) -> {
            if (!channel.type().equals(expr.type())) {
                throw new IllegalArgumentException("channel " + channel + " expects " + channel.type()
                        + " but the surface expression is " + expr.type());
            }
        });
    }

    /** Accumulates fragment statements, binding each intermediate to a fresh local for reuse + readability. */
    private static final class Body {
        private final List<Statement> statements = new ArrayList<>();
        private int counter = 0;

        Expr let(String name, Expr value) {
            LocalVar variable = new LocalVar(name + "_" + counter++, value.type());
            statements.add(new Statement.DeclareVar(variable, value));
            return new Expr.Read(variable);
        }
    }
}
