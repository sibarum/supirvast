package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.tools.GraphicsPipelineSpec.VertexAttribute;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The {@link GraphicsPipelineSpec} contract — the typed description the previewer builds a pipeline from. These
 * are pure-Java structural checks (no GPU/native toolchain): the v1 standard layout, stride/offset arithmetic,
 * and the validation that keeps an unbuildable layout unrepresentable.
 */
class GraphicsPipelineSpecTest {

    private static final byte[] FAKE_VERT = new byte[]{1, 2, 3, 4};   // 4-byte (one-word) stand-ins for SPIR-V
    private static final byte[] FAKE_FRAG = new byte[]{5, 6, 7, 8};
    private static final Type VEC3 = new Type.Vector(Type.float32(), 3);

    @Test
    void standardLayoutIsPositionThenNormalTightlyPacked() {
        GraphicsPipelineSpec spec = GraphicsPipelineSpec.standard(FAKE_VERT, FAKE_FRAG);

        assertEquals("main", spec.vertexEntryPoint());
        assertEquals("main", spec.fragmentEntryPoint());
        assertEquals(2, spec.vertexLayout().size());

        VertexAttribute position = spec.vertexLayout().get(0);
        VertexAttribute normal = spec.vertexLayout().get(1);
        assertEquals("position", position.name());
        assertEquals(0, position.location());
        assertEquals(0, position.offsetBytes());
        assertEquals(12, position.sizeBytes());
        assertEquals("normal", normal.name());
        assertEquals(1, normal.location());
        assertEquals(12, normal.offsetBytes());

        assertEquals(24, spec.vertexStrideBytes());
    }

    @Test
    void rejectsNonContiguousLocations() {
        List<VertexAttribute> layout = List.of(
                new VertexAttribute("position", 0, VEC3, 0),
                new VertexAttribute("normal", 2, VEC3, 12));   // location 2, but position 1
        assertThrows(IllegalArgumentException.class,
                () -> new GraphicsPipelineSpec(FAKE_VERT, "main", FAKE_FRAG, "main", layout));
    }

    @Test
    void rejectsOverlappingAttributes() {
        List<VertexAttribute> layout = List.of(
                new VertexAttribute("position", 0, VEC3, 0),
                new VertexAttribute("normal", 1, VEC3, 8));     // starts at 8, but position ends at 12
        assertThrows(IllegalArgumentException.class,
                () -> new GraphicsPipelineSpec(FAKE_VERT, "main", FAKE_FRAG, "main", layout));
    }

    @Test
    void rejectsUnsupportedAttributeType() {
        assertThrows(IllegalArgumentException.class,
                () -> new VertexAttribute("index", 0, Type.int32(), 0));
        assertThrows(IllegalArgumentException.class,
                () -> new VertexAttribute("huge", 0, new Type.Vector(Type.float32(), 5), 0));
    }

    @Test
    void rejectsEmptyOrMisAlignedSpirv() {
        List<VertexAttribute> layout = List.of(new VertexAttribute("position", 0, VEC3, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new GraphicsPipelineSpec(new byte[0], "main", FAKE_FRAG, "main", layout));
        assertThrows(IllegalArgumentException.class,
                () -> new GraphicsPipelineSpec(new byte[]{1, 2, 3}, "main", FAKE_FRAG, "main", layout));
    }
}
