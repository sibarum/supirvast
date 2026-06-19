package dev.supirvast.vastir.preview;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A minimal Wavefront OBJ loader producing the previewer's {@link Mesh}. Polygons are fan-triangulated and
 * faces expanded to one vertex per corner (no dedup — simple and correct for a PoC). When a face corner has no
 * normal, a flat per-triangle normal is computed from the triangle's positions; missing texcoords default to
 * zero.
 *
 * <p>Supports {@code v}, {@code vn}, {@code vt}, and {@code f} with the {@code p}, {@code p/t}, {@code p//n},
 * and {@code p/t/n} corner forms (positive 1-based indices). Other directives are skipped.
 */
public final class ObjLoader {

    private static final float[] ZERO_UV = {0.0f, 0.0f};

    private ObjLoader() {
    }

    public static Mesh load(Path path) {
        try {
            return parse(Files.readString(path));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read OBJ model: " + path, e);
        }
    }

    public static Mesh parse(String text) {
        List<float[]> positions = new ArrayList<>();
        List<float[]> normals = new ArrayList<>();
        List<float[]> texcoords = new ArrayList<>();
        List<int[]> corners = new ArrayList<>();   // each corner: [posIdx, uvIdxOr-1, normIdxOr-1] (0-based)

        for (String rawLine : text.split("\\R")) {
            String line = rawLine.strip();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            String[] tokens = line.split("\\s+");
            switch (tokens[0]) {
                case "v" -> positions.add(vec3(tokens, line));
                case "vn" -> normals.add(vec3(tokens, line));
                case "vt" -> texcoords.add(vec2(tokens, line));
                case "f" -> triangulate(tokens, corners);
                default -> { /* usemtl, o, g, s, … — ignored for the PoC */ }
            }
        }
        if (corners.isEmpty()) {
            throw new IllegalArgumentException("OBJ has no faces");
        }
        return build(positions, normals, texcoords, corners);
    }

    private static float[] vec3(String[] tokens, String line) {
        if (tokens.length < 4) {
            throw new IllegalArgumentException("expected 3 components in: " + line);
        }
        return new float[]{
                Float.parseFloat(tokens[1]), Float.parseFloat(tokens[2]), Float.parseFloat(tokens[3])};
    }

    private static float[] vec2(String[] tokens, String line) {
        if (tokens.length < 3) {
            throw new IllegalArgumentException("expected 2 components in: " + line);
        }
        return new float[]{Float.parseFloat(tokens[1]), Float.parseFloat(tokens[2])};
    }

    /** Fan-triangulates a face's corners into (0, i, i+1) triangles, recording position/uv/normal indices. */
    private static void triangulate(String[] tokens, List<int[]> out) {
        int cornerCount = tokens.length - 1;
        if (cornerCount < 3) {
            throw new IllegalArgumentException("face needs at least 3 corners: " + String.join(" ", tokens));
        }
        int[][] corners = new int[cornerCount][];
        for (int i = 0; i < cornerCount; i++) {
            corners[i] = parseCorner(tokens[i + 1]);
        }
        for (int i = 1; i + 1 < cornerCount; i++) {
            out.add(corners[0]);
            out.add(corners[i]);
            out.add(corners[i + 1]);
        }
    }

    /** Parses a face corner {@code p}, {@code p/t}, {@code p//n}, or {@code p/t/n} into [posIdx, uvIdx, normIdx]. */
    private static int[] parseCorner(String corner) {
        String[] parts = corner.split("/", -1);
        int position = Integer.parseInt(parts[0]) - 1;
        int uv = parts.length >= 2 && !parts[1].isEmpty() ? Integer.parseInt(parts[1]) - 1 : -1;
        int normal = parts.length >= 3 && !parts[2].isEmpty() ? Integer.parseInt(parts[2]) - 1 : -1;
        return new int[]{position, uv, normal};
    }

    private static Mesh build(List<float[]> positions, List<float[]> normals, List<float[]> texcoords,
            List<int[]> corners) {
        int triangleCount = corners.size() / 3;
        float[] vertices = new float[corners.size() * Mesh.FLOATS_PER_VERTEX];
        int[] indices = new int[corners.size()];

        for (int t = 0; t < triangleCount; t++) {
            int base = t * 3;
            float[] p0 = positions.get(corners.get(base)[0]);
            float[] p1 = positions.get(corners.get(base + 1)[0]);
            float[] p2 = positions.get(corners.get(base + 2)[0]);
            float[] flat = Mesh.flatNormal(p0, p1, p2);

            for (int c = 0; c < 3; c++) {
                int[] corner = corners.get(base + c);
                float[] position = positions.get(corner[0]);
                float[] normal = corner[2] >= 0 ? normals.get(corner[2]) : flat;
                float[] uv = corner[1] >= 0 ? texcoords.get(corner[1]) : ZERO_UV;
                int v = (base + c) * Mesh.FLOATS_PER_VERTEX;
                vertices[v] = position[0];
                vertices[v + 1] = position[1];
                vertices[v + 2] = position[2];
                vertices[v + 3] = normal[0];
                vertices[v + 4] = normal[1];
                vertices[v + 5] = normal[2];
                vertices[v + 6] = uv[0];
                vertices[v + 7] = uv[1];
                indices[base + c] = base + c;
            }
        }
        return new Mesh(vertices, indices);
    }
}
