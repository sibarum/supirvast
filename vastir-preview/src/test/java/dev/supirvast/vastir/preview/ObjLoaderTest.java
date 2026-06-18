package dev.supirvast.vastir.preview;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** OBJ parsing + flat-normal computation — pure logic, no GPU. */
class ObjLoaderTest {

    @Test
    void quadIsTriangulatedToSixCornersWithComputedNormal() {
        // A unit quad in the z=0 plane, wound CCW looking down +z, so the flat normal is +z.
        String obj = """
                v 0 0 0
                v 1 0 0
                v 1 1 0
                v 0 1 0
                f 1 2 3 4
                """;
        Mesh mesh = ObjLoader.parse(obj);

        // One quad → two triangles → six expanded corners, indices 0..5.
        assertEquals(6, mesh.vertexCount());
        assertEquals(6, mesh.indexCount());
        assertEquals(6 * Mesh.FLOATS_PER_VERTEX, mesh.vertices().length);
        for (int i = 0; i < 6; i++) {
            assertEquals(i, mesh.indices()[i]);
        }

        // Every vertex carries the computed +z flat normal (components 3..5 of each 6-float vertex).
        for (int v = 0; v < mesh.vertexCount(); v++) {
            int n = v * Mesh.FLOATS_PER_VERTEX;
            assertEquals(0.0f, mesh.vertices()[n + 3], 1e-6);
            assertEquals(0.0f, mesh.vertices()[n + 4], 1e-6);
            assertEquals(1.0f, mesh.vertices()[n + 5], 1e-6);
        }
    }

    @Test
    void usesSuppliedNormalsWhenPresent() {
        String obj = """
                v 0 0 0
                v 1 0 0
                v 0 1 0
                vn 1 0 0
                f 1//1 2//1 3//1
                """;
        Mesh mesh = ObjLoader.parse(obj);

        assertEquals(3, mesh.vertexCount());
        // Supplied normal (1,0,0) is used verbatim rather than the computed +z face normal.
        assertEquals(1.0f, mesh.vertices()[3], 1e-6);
        assertEquals(0.0f, mesh.vertices()[5], 1e-6);
    }

    @Test
    void skipsCommentsBlankLinesAndUnknownDirectives() {
        String obj = """
                # a triangle
                o thing
                v 0 0 0
                vt 0 0
                v 1 0 0

                v 0 1 0
                f 1/1 2/1 3/1
                """;
        Mesh mesh = ObjLoader.parse(obj);
        assertEquals(3, mesh.vertexCount());
    }

    @Test
    void rejectsModelWithNoFaces() {
        assertThrows(IllegalArgumentException.class, () -> ObjLoader.parse("v 0 0 0\nv 1 0 0\nv 0 1 0\n"));
    }

    @Test
    void parsesPositionsCorrectly() {
        Mesh mesh = ObjLoader.parse("v 2 3 4\nv 5 6 7\nv 8 9 10\nf 1 2 3\n");
        assertTrue(mesh.vertexCount() == 3);
        assertEquals(2.0f, mesh.vertices()[0], 1e-6);
        assertEquals(3.0f, mesh.vertices()[1], 1e-6);
        assertEquals(4.0f, mesh.vertices()[2], 1e-6);
    }
}
