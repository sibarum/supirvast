package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.type.Type;

import java.util.List;

/**
 * What the shader previewer needs to build a graphics pipeline: a vertex + fragment SPIR-V pair (with their
 * entry-point names) and the {@link VertexAttribute vertex layout} the vertex stage reads. This is the typed
 * contract between an authored SupirVast shader and the pipeline that draws with it — the graphics analogue of
 * {@link KernelSpec}, and the previewer's counterpart to {@code KernelColumn}'s ABI.
 *
 * <p>The vertex layout mirrors the shader's location-bound {@code InterfaceVar} inputs (proven expressible by
 * {@code VertexAttributeShaderTest}): each {@link VertexAttribute} names a location, its element {@link Type},
 * and its byte offset within one interleaved vertex. The previewer maps each attribute's {@code Type} to a
 * Vulkan format and the layout to a {@code VkPipelineVertexInputState}; a model loader fills a vertex buffer in
 * this same layout.
 *
 * <p>Attachment formats are intentionally <em>not</em> part of the spec: the color format is dictated by the
 * swapchain surface at runtime, and the depth format is the previewer's choice. The spec describes only what is
 * intrinsic to the shaders.
 *
 * <p>v1 fixes the standard layout via {@link #standard}: location 0 = position {@code vec3}, location 1 =
 * normal {@code vec3}, tightly packed. Build it directly for other layouts as the model formats grow.
 */
public record GraphicsPipelineSpec(
        byte[] vertexSpirv, String vertexEntryPoint,
        byte[] fragmentSpirv, String fragmentEntryPoint,
        List<VertexAttribute> vertexLayout) {

    /**
     * One per-vertex attribute: a {@code location}-bound input of element {@code type} at {@code offsetBytes}
     * within an interleaved vertex. Matches a vertex-stage {@code InterfaceVar.input(name, location, type)}.
     */
    public record VertexAttribute(String name, int location, Type type, int offsetBytes) {
        public VertexAttribute {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("attribute name must be non-blank");
            }
            if (location < 0) {
                throw new IllegalArgumentException("attribute location must be >= 0, got " + location);
            }
            if (offsetBytes < 0) {
                throw new IllegalArgumentException("attribute offset must be >= 0, got " + offsetBytes);
            }
            if (!isSupported(type)) {
                throw new IllegalArgumentException("attribute '" + name + "' is " + type
                        + "; v1 vertex attributes are f32 scalars or f32 vectors (1..4 components)");
            }
        }

        /** Bytes this attribute occupies in a vertex (component width × component count). */
        public int sizeBytes() {
            return byteSize(type);
        }
    }

    public GraphicsPipelineSpec {
        requireSpirv(vertexSpirv, "vertex");
        requireSpirv(fragmentSpirv, "fragment");
        if (vertexEntryPoint == null || vertexEntryPoint.isBlank()) {
            throw new IllegalArgumentException("vertex entry-point name must be non-blank");
        }
        if (fragmentEntryPoint == null || fragmentEntryPoint.isBlank()) {
            throw new IllegalArgumentException("fragment entry-point name must be non-blank");
        }
        if (vertexLayout.isEmpty()) {
            throw new IllegalArgumentException("vertex layout must declare at least one attribute");
        }
        vertexLayout = List.copyOf(vertexLayout);
        // Locations must be 0..n-1 in order (mirrors the contiguous-binding rule for kernel columns), and the
        // attributes must be laid out without overlap so the previewer can pack one interleaved vertex buffer.
        int expectedOffset = 0;
        for (int i = 0; i < vertexLayout.size(); i++) {
            VertexAttribute attribute = vertexLayout.get(i);
            if (attribute.location() != i) {
                throw new IllegalArgumentException("attribute '" + attribute.name() + "' is at position " + i
                        + " but binds location " + attribute.location() + "; locations must be 0..n-1 in order");
            }
            if (attribute.offsetBytes() < expectedOffset) {
                throw new IllegalArgumentException("attribute '" + attribute.name() + "' overlaps the previous "
                        + "attribute (offset " + attribute.offsetBytes() + " < " + expectedOffset + ")");
            }
            expectedOffset = attribute.offsetBytes() + attribute.sizeBytes();
        }
    }

    /** The stride of one interleaved vertex: past the last attribute's end. */
    public int vertexStrideBytes() {
        VertexAttribute last = vertexLayout.get(vertexLayout.size() - 1);
        return last.offsetBytes() + last.sizeBytes();
    }

    /**
     * The standard layout — location 0 = position {@code vec3}, location 1 = normal {@code vec3}, location 2 =
     * uv {@code vec2}, tightly packed (stride 32 bytes) — with the conventional {@code "main"} entry point for
     * both stages. UV is always present so there is a single vertex format; a shader that doesn't read it simply
     * ignores location 2.
     */
    public static GraphicsPipelineSpec standard(byte[] vertexSpirv, byte[] fragmentSpirv) {
        Type vec3 = new Type.Vector(Type.float32(), 3);
        Type vec2 = new Type.Vector(Type.float32(), 2);
        return new GraphicsPipelineSpec(
                vertexSpirv, "main",
                fragmentSpirv, "main",
                List.of(new VertexAttribute("position", 0, vec3, 0),
                        new VertexAttribute("normal", 1, vec3, byteSize(vec3)),
                        new VertexAttribute("uv", 2, vec2, byteSize(vec3) * 2)));
    }

    private static void requireSpirv(byte[] spirv, String stage) {
        if (spirv == null || spirv.length == 0) {
            throw new IllegalArgumentException(stage + " SPIR-V must be non-empty");
        }
        if (spirv.length % 4 != 0) {
            throw new IllegalArgumentException(stage + " SPIR-V length must be a multiple of 4 (it is words), got "
                    + spirv.length);
        }
    }

    private static boolean isSupported(Type type) {
        if (type instanceof Type.Float f) {
            return f.width() == 32;
        }
        return type instanceof Type.Vector v && v.component() instanceof Type.Float f
                && f.width() == 32 && v.count() >= 1 && v.count() <= 4;
    }

    private static int byteSize(Type type) {
        return switch (type) {
            case Type.Float f -> f.width() / 8;
            case Type.Vector v -> byteSize(v.component()) * v.count();
            default -> throw new IllegalArgumentException("no byte size for " + type);
        };
    }
}
