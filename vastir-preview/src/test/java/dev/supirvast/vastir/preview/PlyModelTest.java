package dev.supirvast.vastir.preview;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** PLY parsing — the general channel model (ascii + binary), exposed channels, and the mesh projection. */
class PlyModelTest {

    @Test
    void asciiExposesAllChannelsAndProjectsToMesh() {
        String ply = """
                ply
                format ascii 1.0
                comment a colored quad
                element vertex 4
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                element face 1
                property list uchar int vertex_indices
                end_header
                0 0 0 255 0 0
                1 0 0 0 255 0
                1 1 0 0 0 255
                0 1 0 255 255 0
                4 0 1 2 3
                """;
        PlyModel model = PlyModel.parse(ply.getBytes(StandardCharsets.US_ASCII));

        assertEquals(PlyModel.Format.ASCII, model.format());
        PlyModel.Element vertex = model.element("vertex").orElseThrow();
        assertEquals(4, vertex.count());

        // Every declared channel is parsed and exposed — including the colors the renderer doesn't draw yet.
        assertTrue(vertex.has("red") && vertex.has("green") && vertex.has("blue"));
        assertArrayEquals(new double[]{255, 0, 0, 255}, vertex.scalar("red").orElseThrow());
        assertEquals(0.0, vertex.scalar("x").orElseThrow()[0], 1e-9);
        assertEquals(1.0, vertex.scalar("x").orElseThrow()[1], 1e-9);

        // The face's list property is exposed verbatim (0-based PLY indices).
        int[][] faces = model.element("face").orElseThrow().list("vertex_indices").orElseThrow();
        assertArrayEquals(new int[]{0, 1, 2, 3}, faces[0]);

        // Projection: a quad fan-triangulates to 6 corners with a computed +z flat normal.
        Mesh mesh = model.toMesh();
        assertEquals(6, mesh.vertexCount());
        assertEquals(6, mesh.indexCount());
        assertEquals(1.0f, mesh.vertices()[5], 1e-6);   // first vertex normal z
    }

    @Test
    void usesSuppliedNormalsWhenPresent() {
        String ply = """
                ply
                format ascii 1.0
                element vertex 3
                property float x
                property float y
                property float z
                property float nx
                property float ny
                property float nz
                element face 1
                property list uchar int vertex_indices
                end_header
                0 0 0 1 0 0
                1 0 0 1 0 0
                0 1 0 1 0 0
                3 0 1 2
                """;
        Mesh mesh = PlyModel.parse(ply.getBytes(StandardCharsets.US_ASCII)).toMesh();
        assertEquals(3, mesh.vertexCount());
        // Supplied normal (1,0,0) is used rather than the computed +z face normal.
        assertEquals(1.0f, mesh.vertices()[3], 1e-6);
        assertEquals(0.0f, mesh.vertices()[5], 1e-6);
    }

    @Test
    void parsesBinaryLittleEndian() throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        String header = """
                ply
                format binary_little_endian 1.0
                element vertex 3
                property float x
                property float y
                property float z
                element face 1
                property list uchar int vertex_indices
                end_header
                """;
        out.write(header.getBytes(StandardCharsets.US_ASCII));

        ByteBuffer data = ByteBuffer.allocate(3 * 3 * Float.BYTES + 1 + 3 * Integer.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        float[][] positions = {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}};
        for (float[] p : positions) {
            data.putFloat(p[0]).putFloat(p[1]).putFloat(p[2]);
        }
        data.put((byte) 3).putInt(0).putInt(1).putInt(2);   // one triangle face
        out.write(data.array());

        PlyModel model = PlyModel.parse(out.toByteArray());
        assertEquals(PlyModel.Format.BINARY_LITTLE_ENDIAN, model.format());

        Mesh mesh = model.toMesh();
        assertEquals(3, mesh.vertexCount());
        assertEquals(3, mesh.indexCount());
        assertEquals(2.0f, mesh.vertices()[Mesh.FLOATS_PER_VERTEX], 1e-6);   // 2nd vertex x = 2
    }

    @Test
    void rejectsNonPlyAndMissingVertex() {
        assertThrows(IllegalArgumentException.class,
                () -> PlyModel.parse("not a ply\n".getBytes(StandardCharsets.US_ASCII)));

        String noVertex = """
                ply
                format ascii 1.0
                element thing 1
                property float a
                end_header
                3
                """;
        assertThrows(IllegalArgumentException.class,
                () -> PlyModel.parse(noVertex.getBytes(StandardCharsets.US_ASCII)).toMesh());
    }
}
