package dev.supirvast.vastir.preview;

import java.util.List;

/**
 * A pipeline-ready mesh in the previewer's v1 vertex format: interleaved {@code position(vec3) + normal(vec3)}
 * (24-byte stride, matching {@link dev.supirvast.vastir.tools.GraphicsPipelineSpec#standard}) plus a
 * {@code uint32} index list. Every model loader ({@link ObjLoader}, {@link PlyModel}) produces this so the
 * renderer is format-agnostic.
 */
public record Mesh(float[] vertices, int[] indices) {

    /** Floats per interleaved vertex: position xyz + normal xyz. */
    public static final int FLOATS_PER_VERTEX = 6;

    public int vertexCount() {
        return vertices.length / FLOATS_PER_VERTEX;
    }

    public int indexCount() {
        return indices.length;
    }

    /**
     * Builds a mesh from indexed polygons, fan-triangulated and expanded to one vertex per corner (no dedup).
     * Each polygon is a list of 0-based vertex indices. When {@code normals} is {@code null}, a flat per-triangle
     * normal is computed; otherwise the per-vertex normal at each index is used.
     */
    static Mesh fromPolygons(List<float[]> positions, List<float[]> normals, List<int[]> polygons) {
        int triangleCount = 0;
        for (int[] polygon : polygons) {
            if (polygon.length >= 3) {
                triangleCount += polygon.length - 2;
            }
        }
        float[] vertices = new float[triangleCount * 3 * FLOATS_PER_VERTEX];
        int[] indices = new int[triangleCount * 3];

        int corner = 0;
        for (int[] polygon : polygons) {
            for (int i = 1; i + 1 < polygon.length; i++) {   // fan: (0, i, i+1)
                int a = polygon[0];
                int b = polygon[i];
                int c = polygon[i + 1];
                float[] flat = normals == null
                        ? flatNormal(positions.get(a), positions.get(b), positions.get(c)) : null;
                corner = emit(vertices, indices, corner, positions, normals, flat, a);
                corner = emit(vertices, indices, corner, positions, normals, flat, b);
                corner = emit(vertices, indices, corner, positions, normals, flat, c);
            }
        }
        return new Mesh(vertices, indices);
    }

    private static int emit(float[] vertices, int[] indices, int corner, List<float[]> positions,
            List<float[]> normals, float[] flat, int index) {
        float[] position = positions.get(index);
        float[] normal = normals != null ? normals.get(index) : flat;
        int v = corner * FLOATS_PER_VERTEX;
        vertices[v] = position[0];
        vertices[v + 1] = position[1];
        vertices[v + 2] = position[2];
        vertices[v + 3] = normal[0];
        vertices[v + 4] = normal[1];
        vertices[v + 5] = normal[2];
        indices[corner] = corner;
        return corner + 1;
    }

    /** Unit-length normal of the triangle (p1-p0) × (p2-p0); a degenerate triangle yields (0,0,1). */
    static float[] flatNormal(float[] p0, float[] p1, float[] p2) {
        float ax = p1[0] - p0[0];
        float ay = p1[1] - p0[1];
        float az = p1[2] - p0[2];
        float bx = p2[0] - p0[0];
        float by = p2[1] - p0[1];
        float bz = p2[2] - p0[2];
        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;
        float length = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (length == 0.0f) {
            return new float[]{0.0f, 0.0f, 1.0f};
        }
        return new float[]{nx / length, ny / length, nz / length};
    }
}
