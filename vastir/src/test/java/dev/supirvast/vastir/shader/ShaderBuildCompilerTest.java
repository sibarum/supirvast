package dev.supirvast.vastir.shader;

import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ShaderBuildCompilerTest {

    private static final int SPIRV_MAGIC = 0x0723_0203;

    /** {@code fragColor = vec4(1, 0, 1, 1)} — the smallest fragment shader the IR can express. */
    public static final class MagentaFragment implements ShaderSource {
        @Override
        public String name() {
            return "test-magenta.frag";
        }

        @Override
        public CoreModule module() {
            Type.Float f32 = Type.float32();
            Type.Vector vec4 = new Type.Vector(f32, 4);
            InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, vec4);
            Region body = Region.of(
                    new Statement.InterfaceWrite(fragColor, new Expr.VectorConstruct(vec4, List.of(
                            new Expr.ConstFloat(f32, 1.0), new Expr.ConstFloat(f32, 0.0),
                            new Expr.ConstFloat(f32, 1.0), new Expr.ConstFloat(f32, 1.0)))),
                    new Statement.ReturnVoid());
            Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
            return new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT));
        }
    }

    @Test
    void compilesDiscoveredSourcesAndLoadsThemBack(@TempDir Path out) throws Exception {
        Path testClasses = Path.of(MagentaFragment.class.getProtectionDomain()
                .getCodeSource().getLocation().toURI());

        List<String> names = ShaderBuildCompiler.compile(testClasses, out);
        assertTrue(names.contains("test-magenta.frag"), "discovered names: " + names);

        Path spv = out.resolve(Shaders.RESOURCE_DIR).resolve("test-magenta.frag.spv");
        byte[] bytes = Files.readAllBytes(spv);
        assertEquals(SPIRV_MAGIC, magic(bytes));

        // The runtime loader resolves the compiled binary as a classpath resource — no re-lowering.
        try (URLClassLoader loader = new URLClassLoader(new URL[]{out.toUri().toURL()}, null)) {
            assertTrue(Shaders.isPrecompiled("test-magenta.frag", loader));
            assertArrayEquals(bytes, Shaders.load("test-magenta.frag", loader));
            assertTrue(Shaders.names(loader).contains("test-magenta.frag"));
        }

        // Recompiling into the same directory is a no-op (bytes unchanged → file untouched).
        var before = Files.getLastModifiedTime(spv);
        ShaderBuildCompiler.compile(testClasses, out);
        assertEquals(before, Files.getLastModifiedTime(spv));
    }

    @Test
    void loadOrLowerFallsBackToLoweringWithoutTheResource() {
        byte[] lowered = Shaders.loadOrLower(new MagentaFragment());
        assertEquals(SPIRV_MAGIC, magic(lowered));
    }

    @Test
    void loadThrowsForUnknownName() throws Exception {
        try (URLClassLoader empty = new URLClassLoader(new URL[0], null)) {
            assertFalse(Shaders.isPrecompiled("nope", empty));
            assertThrows(IllegalArgumentException.class, () -> Shaders.load("nope", empty));
        }
    }

    private static int magic(byte[] spirv) {
        return (spirv[0] & 0xFF) | (spirv[1] & 0xFF) << 8 | (spirv[2] & 0xFF) << 16 | (spirv[3] & 0xFF) << 24;
    }
}
