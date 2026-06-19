package dev.supirvast.studio;

/**
 * A CPU-side triangle mesh: an interleaved vertex buffer and a triangle index
 * buffer, in exactly the layout the studio's fixed vertex shader expects.
 *
 * <p>Each vertex is eight floats — {@code position.xyz}, {@code normal.xyz},
 * {@code uv.xy} — matching the {@code (location = 0..2)} attribute declarations
 * in {@code shaders/model.vert}. {@link #STRIDE_FLOATS} is the per-vertex
 * stride a {@code glVertexAttribPointer} setup walks.
 *
 * @param vertices interleaved {@code pos3 normal3 uv2} data
 * @param indices  triangle list into {@link #vertices}
 */
public record Geometry(float[] vertices, int[] indices) {

    /** Floats per vertex: 3 position + 3 normal + 2 uv. */
    public static final int STRIDE_FLOATS = 8;

    /** Number of triangle indices to draw. */
    public int indexCount() {
        return indices.length;
    }

    /**
     * Generate a unit-radius UV sphere centred at the origin. Normals are the
     * unit position (a sphere's defining property), and uv is the standard
     * longitude/latitude parameterisation so the editor's fragment shader can
     * key surface detail off either.
     *
     * @param stacks latitude divisions (rings), must be &ge; 2
     * @param slices longitude divisions (segments per ring), must be &ge; 3
     */
    public static Geometry uvSphere(int stacks, int slices) {
        if (stacks < 2) throw new IllegalArgumentException("stacks must be >= 2");
        if (slices < 3) throw new IllegalArgumentException("slices must be >= 3");

        int vertexCount = (stacks + 1) * (slices + 1);
        float[] vertices = new float[vertexCount * STRIDE_FLOATS];
        int v = 0;
        for (int i = 0; i <= stacks; i++) {
            // phi: 0 at the north pole, PI at the south pole.
            float phi = (float) (Math.PI * i / stacks);
            float sinPhi = (float) Math.sin(phi);
            float cosPhi = (float) Math.cos(phi);
            for (int j = 0; j <= slices; j++) {
                float theta = (float) (2.0 * Math.PI * j / slices);
                float sinTheta = (float) Math.sin(theta);
                float cosTheta = (float) Math.cos(theta);

                float x = sinPhi * cosTheta;
                float y = cosPhi;
                float z = sinPhi * sinTheta;

                // position
                vertices[v++] = x;
                vertices[v++] = y;
                vertices[v++] = z;
                // normal (unit position for a unit sphere)
                vertices[v++] = x;
                vertices[v++] = y;
                vertices[v++] = z;
                // uv
                vertices[v++] = (float) j / slices;
                vertices[v++] = (float) i / stacks;
            }
        }

        int[] indices = new int[stacks * slices * 6];
        int n = 0;
        int rowStride = slices + 1;
        for (int i = 0; i < stacks; i++) {
            for (int j = 0; j < slices; j++) {
                int topLeft = i * rowStride + j;
                int bottomLeft = topLeft + rowStride;
                int topRight = topLeft + 1;
                int bottomRight = bottomLeft + 1;

                indices[n++] = topLeft;
                indices[n++] = bottomLeft;
                indices[n++] = topRight;

                indices[n++] = topRight;
                indices[n++] = bottomLeft;
                indices[n++] = bottomRight;
            }
        }
        return new Geometry(vertices, indices);
    }
}
