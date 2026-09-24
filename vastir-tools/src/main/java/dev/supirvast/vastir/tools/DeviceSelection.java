package dev.supirvast.vastir.tools;

import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Which physical device a {@link GpuContext} runs on.
 *
 * <p>By default the discrete GPU, then an integrated one, then anything else with a compute queue; ties go to
 * enumeration order. The default used to be simply the first device with a compute queue, which on a laptop
 * is usually the integrated GPU — chosen by the order the loader happens to list devices in, and not what a
 * renderer on the same machine picks (VexelRay's prefers the discrete GPU too).
 *
 * <p>The {@value #PROPERTY} system property overrides it: {@code discrete} or {@code integrated} for a device
 * type, or any other text to match a device name case-insensitively ({@code -Dsupirvast.gpu=intel}). An
 * override that matches nothing is an error naming the devices there are, never a silent fallback: a
 * benchmark asked to run on one GPU must not quietly run on another.
 */
final class DeviceSelection {

    /** The system property that overrides the choice. */
    static final String PROPERTY = "supirvast.gpu";

    // VkPhysicalDeviceType values.
    static final int OTHER = 0;
    static final int INTEGRATED = 1;
    static final int DISCRETE = 2;
    static final int VIRTUAL = 3;
    static final int CPU = 4;

    /** A physical device as enumerated: its position, name, type, and whether it has a compute queue. */
    record Candidate(int index, String name, int type, boolean compute) {}

    private DeviceSelection() {
    }

    /**
     * The index of the device to use among {@code candidates}, or -1 if none has a compute queue.
     *
     * @param selector the {@value #PROPERTY} override, or null / blank for the default
     * @throws IllegalStateException if a selector is given and no compute device matches it
     */
    static int choose(List<Candidate> candidates, String selector) {
        List<Candidate> compute = candidates.stream().filter(Candidate::compute).toList();
        if (selector == null || selector.isBlank()) {
            return compute.stream()
                    .min(Comparator.comparingInt((Candidate c) -> rank(c.type())).thenComparingInt(Candidate::index))
                    .map(Candidate::index)
                    .orElse(-1);
        }
        String wanted = selector.trim().toLowerCase(Locale.ROOT);
        return compute.stream()
                .filter(c -> switch (wanted) {
                    case "discrete" -> c.type() == DISCRETE;
                    case "integrated" -> c.type() == INTEGRATED;
                    default -> c.name().toLowerCase(Locale.ROOT).contains(wanted);
                })
                .findFirst()
                .map(Candidate::index)
                .orElseThrow(() -> new IllegalStateException("-D" + PROPERTY + "=" + selector
                        + " matches no Vulkan compute device; there are: " + compute.stream()
                        .map(c -> c.name() + " (" + typeName(c.type()) + ")").collect(Collectors.joining(", "))));
    }

    /** The system property's value, read when a context is opened. */
    static String selector() {
        return System.getProperty(PROPERTY);
    }

    static String typeName(int type) {
        return switch (type) {
            case INTEGRATED -> "integrated";
            case DISCRETE -> "discrete";
            case VIRTUAL -> "virtual";
            case CPU -> "cpu";
            default -> "other";
        };
    }

    private static int rank(int type) {
        return switch (type) {
            case DISCRETE -> 0;
            case INTEGRATED -> 1;
            case VIRTUAL -> 2;
            case CPU -> 3;
            default -> 4;
        };
    }
}
