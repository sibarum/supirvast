package dev.supirvast.vastir.pbr;

import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.tools.GraphicsPipelineSpec;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static dev.supirvast.vastir.pbr.Shade.f;
import static dev.supirvast.vastir.pbr.Shade.sample;
import static dev.supirvast.vastir.pbr.Shade.vec3;
import static dev.supirvast.vastir.pbr.Shade.xyz;

/**
 * Generates a renderable PBR demo: Cook-Torrance shaders from {@link PbrShader} plus a UV-sphere model (a
 * curved surface shows roughness/metallic and the specular highlight far better than a flat-faced cube). Run
 * it, then point the previewer at the outputs:
 *
 * <pre>java dev.supirvast.vastir.pbr.PbrSampleAssets &lt;outDir&gt;</pre>
 *
 * <p>Writes a shared {@code sphere.obj} + {@code pbr.vert.spv}, and one {@code <name>.frag.spv} per demo
 * material. The materials span the two PBR axes: a smooth and a rough red dielectric (strong diffuse + a white
 * highlight) and a copper metal (no diffuse — dark body with a colored highlight, since there's no environment
 * lighting yet). The sphere is authored in clip space (no projection) for the static previewer; vertex normals
 * are the geometric sphere normals.
 */
public final class PbrSampleAssets {

    private PbrSampleAssets() {
    }

    public static void main(String[] args) {
        Path outDir = Path.of(args.length > 0 ? args[0] : "pbr-samples");
        try {
            Files.createDirectories(outDir);
            Files.writeString(outDir.resolve("sphere.obj"), sphereObj(32, 48, 0.45f, 0.5f));
            boolean wroteVertex = false;
            for (Map.Entry<String, PbrShader> material : materials().entrySet()) {
                GraphicsPipelineSpec spec = material.getValue().spec();
                if (!wroteVertex) {   // the vertex stage is identical across materials
                    Files.write(outDir.resolve("pbr.vert.spv"), spec.vertexSpirv());
                    wroteVertex = true;
                }
                Files.write(outDir.resolve(material.getKey() + ".frag.spv"), spec.fragmentSpirv());
            }
            // MVP demo: a textured PBR material that transforms by a push-constant MVP and lights in world
            // space, plus a model-space sphere. Render with --mvp --texture 0=<checker>.
            GraphicsPipelineSpec mvpSpec = textured().withMvp().spec();
            Files.write(outDir.resolve("pbr-mvp.vert.spv"), mvpSpec.vertexSpirv());
            Files.write(outDir.resolve("textured-mvp.frag.spv"), mvpSpec.fragmentSpirv());
            Files.writeString(outDir.resolve("sphere-model.obj"), sphereObj(32, 48, 1.0f, 0.0f));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to write PBR sample assets to " + outDir, e);
        }
        System.out.println("[pbr-sample] wrote sphere.obj, sphere-model.obj, pbr.vert.spv, pbr-mvp.{vert,frag}, "
                + "and " + materials().keySet() + " fragment shaders to " + outDir);
    }

    /** The demo materials, spanning roughness and metalness. */
    static Map<String, PbrShader> materials() {
        Map<String, PbrShader> out = new LinkedHashMap<>();
        out.put("dielectric-smooth", dielectric(vec3(0.85, 0.13, 0.1), 0.2));
        out.put("dielectric-rough", dielectric(vec3(0.85, 0.13, 0.1), 0.6));
        out.put("metal-copper", metal(vec3(0.95, 0.64, 0.54), 0.3));
        out.put("textured", textured());
        out.put("ibl-metal", iblMetal());
        return out;
    }

    /** A polished metal lit by an environment cubemap (set 0, binding 0) — reflects instead of going dark. */
    private static PbrShader iblMetal() {
        return PbrShader.createWithEnvironment(
                Set.of(Channel.ALBEDO, Channel.METALLIC, Channel.ROUGHNESS),
                inputs -> Map.of(
                        Channel.ALBEDO, vec3(0.95, 0.85, 0.55),   // gold-ish
                        Channel.METALLIC, f(1.0),
                        Channel.ROUGHNESS, f(0.15)),
                0);
    }

    /** A dielectric whose albedo is sampled from a texture (descriptor set 0, binding 0) at the surface UV. */
    private static PbrShader textured() {
        return PbrShader.create(
                Set.of(Channel.ALBEDO, Channel.METALLIC, Channel.ROUGHNESS),
                inputs -> Map.of(
                        Channel.ALBEDO, xyz(sample("albedo", 0, inputs.uv())),
                        Channel.METALLIC, f(0.0),
                        Channel.ROUGHNESS, f(0.4)));
    }

    private static PbrShader dielectric(Expr albedo, double roughness) {
        return PbrShader.create(
                Set.of(Channel.ALBEDO, Channel.METALLIC, Channel.ROUGHNESS),
                inputs -> Map.of(
                        Channel.ALBEDO, albedo,
                        Channel.METALLIC, f(0.0),
                        Channel.ROUGHNESS, f(roughness)));
    }

    private static PbrShader metal(Expr albedo, double roughness) {
        return PbrShader.create(
                Set.of(Channel.ALBEDO, Channel.METALLIC, Channel.ROUGHNESS),
                inputs -> Map.of(
                        Channel.ALBEDO, albedo,
                        Channel.METALLIC, f(1.0),
                        Channel.ROUGHNESS, f(roughness)));
    }

    /**
     * A UV sphere as Wavefront OBJ with explicit normals and texcoords, centered at {@code (0,0,zCenter)} in
     * clip space. Normals are the unit sphere directions; positions are {@code center + radius*normal}; UVs are
     * {@code (longitude, latitude)} in [0,1].
     */
    static String sphereObj(int stacks, int slices, float radius, float zCenter) {
        StringBuilder obj = new StringBuilder("# generated by PbrSampleAssets — UV sphere in clip space\n");
        int stride = slices + 1;
        for (int i = 0; i <= stacks; i++) {
            double theta = Math.PI * i / stacks;            // 0 (top) .. PI (bottom)
            double sinTheta = Math.sin(theta);
            double cosTheta = Math.cos(theta);
            for (int j = 0; j <= slices; j++) {
                double phi = 2.0 * Math.PI * j / slices;
                float nx = (float) (sinTheta * Math.cos(phi));
                float ny = (float) cosTheta;
                float nz = (float) (sinTheta * Math.sin(phi));
                obj.append("v ").append(radius * nx).append(' ').append(radius * ny).append(' ')
                        .append(zCenter + radius * nz).append('\n');
                obj.append("vt ").append((float) j / slices).append(' ').append((float) i / stacks).append('\n');
                obj.append("vn ").append(nx).append(' ').append(ny).append(' ').append(nz).append('\n');
            }
        }
        // Quad faces between adjacent rings/segments (1-based; v/vt/vn indices coincide). Pole quads degenerate
        // harmlessly into triangles.
        for (int i = 0; i < stacks; i++) {
            for (int j = 0; j < slices; j++) {
                int a = i * stride + j + 1;
                int b = a + 1;
                int c = a + stride;
                int d = c + 1;
                obj.append("f ").append(corner(a)).append(' ').append(corner(b)).append(' ')
                        .append(corner(d)).append(' ').append(corner(c)).append('\n');
            }
        }
        return obj.toString();
    }

    /** A face corner {@code i/i/i} (position/texcoord/normal share the same index in this grid). */
    private static String corner(int i) {
        return i + "/" + i + "/" + i;
    }
}
