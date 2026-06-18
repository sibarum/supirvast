package dev.supirvast.vastir.preview;

import java.nio.file.Path;
import java.util.Locale;

/**
 * Dispatches model loading by file extension, returning the previewer's format-agnostic {@link Mesh}.
 * {@code .obj} → {@link ObjLoader}; {@code .ply} → {@link PlyModel} (projected via {@link PlyModel#toMesh()}).
 */
public final class ModelLoader {

    private ModelLoader() {
    }

    public static Mesh load(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".obj")) {
            return ObjLoader.load(path);
        }
        if (name.endsWith(".ply")) {
            return PlyModel.load(path).toMesh();
        }
        throw new IllegalArgumentException("unsupported model format (expected .obj or .ply): " + path);
    }
}
