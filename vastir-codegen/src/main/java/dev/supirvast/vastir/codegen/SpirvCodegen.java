package dev.supirvast.vastir.codegen;

import com.google.gson.Gson;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Entry point for build-time SPIR-V code generation.
 *
 * <p>Reads the vendored, version-pinned grammar from the classpath and emits Java sources under the output
 * root passed as {@code args[0]} (wired from Maven as {@code target/generated-sources/spirv}). Run as a
 * developer/build tool only; never on the {@code vastir} runtime classpath.
 */
public final class SpirvCodegen {

    /** Classpath location of the vendored grammar (see vastir-codegen/src/main/resources). */
    private static final String GRAMMAR_RESOURCE = "/spirv/unified1/spirv.core.grammar.json";

    /** Package the generated SPIR-V types live in, within the vastir module. */
    static final String GENERATED_PACKAGE = "dev.supirvast.vastir.spirv";

    private SpirvCodegen() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            throw new IllegalArgumentException("usage: SpirvCodegen <output-source-root>");
        }
        Path outputRoot = Path.of(args[0]);
        Grammar grammar = loadGrammar();

        System.out.printf(
                "[vastir-codegen] SPIR-V %s rev %d: %d instructions, %d operand kinds%n",
                grammar.spirvVersion(), grammar.revision,
                grammar.instructions.size(), grammar.operandKinds.size());

        List<Path> written = new ArrayList<>();
        written.addAll(new SharedTypesGenerator(grammar).generateTo(outputRoot));
        written.add(new OpEnumGenerator(grammar).generateTo(outputRoot));
        written.addAll(new OperandEnumGenerator(grammar).generateTo(outputRoot));

        System.out.printf("[vastir-codegen] generated %d Java source files%n", written.size());
    }

    private static Grammar loadGrammar() {
        try (InputStream in = SpirvCodegen.class.getResourceAsStream(GRAMMAR_RESOURCE)) {
            if (in == null) {
                throw new IllegalStateException("grammar resource not found: " + GRAMMAR_RESOURCE);
            }
            try (Reader reader = new InputStreamReader(in, StandardCharsets.UTF_8)) {
                return new Gson().fromJson(reader, Grammar.class);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read SPIR-V grammar", e);
        }
    }

    /** Resolves the .java file path for a generated type and ensures its parent directories exist. */
    static Path javaFile(Path outputRoot, String simpleName) throws IOException {
        Path dir = outputRoot.resolve(GENERATED_PACKAGE.replace('.', '/'));
        Files.createDirectories(dir);
        return dir.resolve(simpleName + ".java");
    }
}
