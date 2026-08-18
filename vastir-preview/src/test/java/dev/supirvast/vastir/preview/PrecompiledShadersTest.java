package dev.supirvast.vastir.preview;

import dev.supirvast.vastir.shader.Shaders;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies the supirvast-maven-plugin ran during this module's build: the model shader pair must be present
 * as pre-compiled classpath resources, identical to what in-process lowering would produce.
 */
class PrecompiledShadersTest {

    @Test
    void modelShaderPairIsPrecompiledOnTheClasspath() {
        ClassLoader loader = ModelShaders.class.getClassLoader();
        assertTrue(Shaders.isPrecompiled("model.vert", loader),
                "model.vert.spv missing — did the compile-shaders goal run?");
        assertTrue(Shaders.isPrecompiled("model.frag", loader));
        assertTrue(Shaders.names(loader).containsAll(java.util.List.of("model.vert", "model.frag")));
    }

    @Test
    void precompiledBinaryMatchesInProcessLowering() {
        var vertex = new ModelShaders.Vertex();
        assertArrayEquals(
                new dev.supirvast.vastir.lower.CoreToSpirv()
                        .lower(vertex.module(), vertex.target()).toByteArray(),
                Shaders.load("model.vert", ModelShaders.class.getClassLoader()));
    }
}
