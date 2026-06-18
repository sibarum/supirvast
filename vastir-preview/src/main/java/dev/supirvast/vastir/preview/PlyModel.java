package dev.supirvast.vastir.preview;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A PLY (Polygon File Format / Stanford) parser for the general element/property model: the header declares
 * named {@link Element elements} (e.g. {@code vertex}, {@code face}, or any user-defined name), each with named
 * scalar {@link Property properties} ({@code x}, {@code nx}, {@code red}, {@code s}, …) and optional list
 * properties ({@code vertex_indices}). Every declared channel is parsed and exposed, regardless of whether the
 * renderer uses it — that open-ended channel model is PLY's distinguishing feature.
 *
 * <p>Supports {@code ascii}, {@code binary_little_endian}, and {@code binary_big_endian}. {@link #toMesh()}
 * projects the parsed data onto the previewer's position+normal {@link Mesh}: it reads {@code x/y/z} (and
 * {@code nx/ny/nz} if present, else a computed flat normal) from the {@code vertex} element and the first list
 * property of the {@code face} element as polygons. Other channels (colors, UVs, custom) are available via
 * {@link #element} but not yet drawn.
 */
public final class PlyModel {

    public enum Format { ASCII, BINARY_LITTLE_ENDIAN, BINARY_BIG_ENDIAN }

    /** A PLY scalar type, with its byte width and how to read it (always normalized to {@code double}). */
    public enum PlyType {
        CHAR(1), UCHAR(1), SHORT(2), USHORT(2), INT(4), UINT(4), FLOAT(4), DOUBLE(8);

        final int bytes;

        PlyType(int bytes) {
            this.bytes = bytes;
        }

        static PlyType of(String token) {
            return switch (token) {
                case "char", "int8" -> CHAR;
                case "uchar", "uint8" -> UCHAR;
                case "short", "int16" -> SHORT;
                case "ushort", "uint16" -> USHORT;
                case "int", "int32" -> INT;
                case "uint", "uint32" -> UINT;
                case "float", "float32" -> FLOAT;
                case "double", "float64" -> DOUBLE;
                default -> throw new IllegalArgumentException("unknown PLY type: " + token);
            };
        }

        double read(ByteBuffer buffer) {
            return switch (this) {
                case CHAR -> buffer.get();
                case UCHAR -> buffer.get() & 0xFF;
                case SHORT -> buffer.getShort();
                case USHORT -> buffer.getShort() & 0xFFFF;
                case INT -> buffer.getInt();
                case UINT -> buffer.getInt() & 0xFFFFFFFFL;
                case FLOAT -> buffer.getFloat();
                case DOUBLE -> buffer.getDouble();
            };
        }
    }

    /** A property declaration: a scalar of {@code valueType}, or a list (count of {@code countType}). */
    public record Property(String name, boolean list, PlyType countType, PlyType valueType) {}

    /** One parsed element: its declared properties plus the values read for each (scalars and lists). */
    public static final class Element {
        private final String name;
        private final int count;
        private final List<Property> properties;
        private final Map<String, double[]> scalars = new LinkedHashMap<>();   // property -> count values
        private final Map<String, int[][]> lists = new LinkedHashMap<>();      // property -> per-record indices

        Element(String name, int count, List<Property> properties) {
            this.name = name;
            this.count = count;
            this.properties = List.copyOf(properties);
        }

        public String name() {
            return name;
        }

        public int count() {
            return count;
        }

        public List<Property> properties() {
            return properties;
        }

        public boolean has(String property) {
            return scalars.containsKey(property) || lists.containsKey(property);
        }

        /** The per-record values of a scalar property (length {@link #count}), or empty if not a scalar here. */
        public Optional<double[]> scalar(String property) {
            return Optional.ofNullable(scalars.get(property));
        }

        /** The per-record index arrays of a list property, or empty if not a list here. */
        public Optional<int[][]> list(String property) {
            return Optional.ofNullable(lists.get(property));
        }

        Optional<int[][]> firstList() {
            return lists.values().stream().findFirst();
        }
    }

    private final Format format;
    private final List<Element> elements;

    private PlyModel(Format format, List<Element> elements) {
        this.format = format;
        this.elements = elements;
    }

    public Format format() {
        return format;
    }

    public List<Element> elements() {
        return elements;
    }

    public Optional<Element> element(String name) {
        return elements.stream().filter(e -> e.name.equals(name)).findFirst();
    }

    public static PlyModel load(Path path) {
        try {
            return parse(Files.readAllBytes(path));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read PLY model: " + path, e);
        }
    }

    public static PlyModel parse(byte[] bytes) {
        Header header = parseHeader(bytes);
        if (header.format == Format.ASCII) {
            return readAscii(header, bytes);
        }
        return readBinary(header, bytes);
    }

    /** Projects the parsed channels onto the previewer's position+normal mesh. */
    public Mesh toMesh() {
        Element vertex = element("vertex")
                .orElseThrow(() -> new IllegalArgumentException("PLY has no 'vertex' element"));
        double[] xs = required(vertex, "x");
        double[] ys = required(vertex, "y");
        double[] zs = required(vertex, "z");

        List<float[]> positions = new ArrayList<>(vertex.count);
        for (int i = 0; i < vertex.count; i++) {
            positions.add(new float[]{(float) xs[i], (float) ys[i], (float) zs[i]});
        }

        List<float[]> normals = null;
        if (vertex.has("nx") && vertex.has("ny") && vertex.has("nz")) {
            double[] nx = vertex.scalars.get("nx");
            double[] ny = vertex.scalars.get("ny");
            double[] nz = vertex.scalars.get("nz");
            normals = new ArrayList<>(vertex.count);
            for (int i = 0; i < vertex.count; i++) {
                normals.add(new float[]{(float) nx[i], (float) ny[i], (float) nz[i]});
            }
        }

        int[][] faces = element("face").flatMap(Element::firstList)
                .or(() -> elements.stream().map(Element::firstList).filter(Optional::isPresent)
                        .map(Optional::get).findFirst())
                .orElseThrow(() -> new IllegalArgumentException("PLY has no face list property"));

        List<int[]> polygons = List.of(faces);
        return Mesh.fromPolygons(positions, normals, polygons);
    }

    private static double[] required(Element element, String property) {
        return element.scalar(property)
                .orElseThrow(() -> new IllegalArgumentException(
                        "vertex element is missing required property '" + property + "'"));
    }

    // --- header ----------------------------------------------------------------------------------------

    private record Header(Format format, List<Element> elements, int dataOffset) {}

    private static Header parseHeader(byte[] bytes) {
        Format format = null;
        List<Element> elements = new ArrayList<>();
        List<Property> currentProps = null;
        String currentName = null;
        int currentCount = 0;

        int pos = 0;
        boolean sawPly = false;
        boolean ended = false;
        while (pos < bytes.length) {
            int newline = indexOf(bytes, (byte) '\n', pos);
            if (newline < 0) {
                break;
            }
            String line = new String(bytes, pos, newline - pos, StandardCharsets.US_ASCII).strip();
            pos = newline + 1;
            if (line.isEmpty()) {
                continue;
            }
            String[] tokens = line.split("\\s+");
            switch (tokens[0]) {
                case "ply" -> sawPly = true;
                case "comment", "obj_info" -> { /* ignored */ }
                case "format" -> format = parseFormat(tokens[1]);
                case "element" -> {
                    if (currentProps != null) {
                        elements.add(new Element(currentName, currentCount, currentProps));
                    }
                    currentName = tokens[1];
                    currentCount = Integer.parseInt(tokens[2]);
                    currentProps = new ArrayList<>();
                }
                case "property" -> currentProps.add(parseProperty(tokens));
                case "end_header" -> ended = true;
                default -> throw new IllegalArgumentException("unexpected PLY header line: " + line);
            }
            if (ended) {
                break;
            }
        }
        if (!sawPly) {
            throw new IllegalArgumentException("not a PLY file (missing 'ply' magic)");
        }
        if (!ended) {
            throw new IllegalArgumentException("PLY header has no end_header");
        }
        if (format == null) {
            throw new IllegalArgumentException("PLY header has no format line");
        }
        if (currentProps != null) {
            elements.add(new Element(currentName, currentCount, currentProps));
        }
        return new Header(format, elements, pos);
    }

    private static Format parseFormat(String token) {
        return switch (token) {
            case "ascii" -> Format.ASCII;
            case "binary_little_endian" -> Format.BINARY_LITTLE_ENDIAN;
            case "binary_big_endian" -> Format.BINARY_BIG_ENDIAN;
            default -> throw new IllegalArgumentException("unknown PLY format: " + token);
        };
    }

    private static Property parseProperty(String[] tokens) {
        if (tokens[1].equals("list")) {
            // property list <countType> <valueType> <name>
            return new Property(tokens[4], true, PlyType.of(tokens[2]), PlyType.of(tokens[3]));
        }
        // property <type> <name>
        return new Property(tokens[2], false, null, PlyType.of(tokens[1]));
    }

    // --- data ------------------------------------------------------------------------------------------

    private static PlyModel readAscii(Header header, byte[] bytes) {
        String data = new String(bytes, header.dataOffset, bytes.length - header.dataOffset,
                StandardCharsets.US_ASCII);
        String[] tokens = data.isBlank() ? new String[0] : data.trim().split("\\s+");
        int[] cursor = {0};
        for (Element element : header.elements) {
            allocate(element);
            for (int record = 0; record < element.count; record++) {
                for (Property property : element.properties) {
                    if (property.list) {
                        int n = (int) Double.parseDouble(tokens[cursor[0]++]);
                        int[] values = new int[n];
                        for (int i = 0; i < n; i++) {
                            values[i] = (int) Double.parseDouble(tokens[cursor[0]++]);
                        }
                        element.lists.get(property.name)[record] = values;
                    } else {
                        element.scalars.get(property.name)[record] = Double.parseDouble(tokens[cursor[0]++]);
                    }
                }
            }
        }
        return new PlyModel(header.format, header.elements);
    }

    private static PlyModel readBinary(Header header, byte[] bytes) {
        ByteOrder order = header.format == Format.BINARY_LITTLE_ENDIAN
                ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN;
        ByteBuffer buffer = ByteBuffer.wrap(bytes, header.dataOffset, bytes.length - header.dataOffset)
                .order(order);
        for (Element element : header.elements) {
            allocate(element);
            for (int record = 0; record < element.count; record++) {
                for (Property property : element.properties) {
                    if (property.list) {
                        int n = (int) property.countType.read(buffer);
                        int[] values = new int[n];
                        for (int i = 0; i < n; i++) {
                            values[i] = (int) property.valueType.read(buffer);
                        }
                        element.lists.get(property.name)[record] = values;
                    } else {
                        element.scalars.get(property.name)[record] = property.valueType.read(buffer);
                    }
                }
            }
        }
        return new PlyModel(header.format, header.elements);
    }

    private static void allocate(Element element) {
        for (Property property : element.properties) {
            if (property.list) {
                element.lists.put(property.name, new int[element.count][]);
            } else {
                element.scalars.put(property.name, new double[element.count]);
            }
        }
    }

    private static int indexOf(byte[] bytes, byte target, int from) {
        for (int i = from; i < bytes.length; i++) {
            if (bytes[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
