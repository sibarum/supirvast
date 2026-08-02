package dev.supirvast.vastir.preview;

import dev.supirvast.vastir.tools.Fullscreen;

/**
 * Spike for the Orchestration API's Conductor: one main thread driving two differently-cadenced Players with no
 * coupling between them — a Vulkan render window (eager, vsync-paced) and a slower "logic" concern (every
 * 500 ms). This is the separation-of-concerns the design is for: the render loop no longer owns the thread, it
 * is just one Player the Conductor ticks alongside others.
 *
 * <pre>java dev.supirvast.vastir.preview.ConductorDemo [renderFrames]</pre>
 */
public final class ConductorDemo {

    private ConductorDemo() {
    }

    public static void main(String[] args) {
        int renderFrames = args.length > 0 ? Integer.parseInt(args[0]) : 180;
        byte[] vert = Fullscreen.triangleVertexSpirv();
        byte[] frag = Fullscreen.constantColorFragmentSpirv(0.11, 0.22, 0.44, 1.0);
        try (WindowedVulkanContext window = new WindowedVulkanContext(
                "Conductor — render + logic", 640, 400, vert, Fullscreen.ENTRY_POINT, frag, Fullscreen.ENTRY_POINT)) {

            // The render Player: eager, ticks one frame each pass; retires after renderFrames (or window close).
            int[] frame = {0};
            Player render = now -> window.tick() && ++frame[0] < renderFrames;

            // The logic Player: a slower, entirely independent concern on a 500 ms cadence.
            long start = System.nanoTime();
            int[] beat = {0};
            Player logic = now -> {
                long ms = (now - start) / 1_000_000L;
                System.out.println("[conductor] logic beat " + (++beat[0]) + " at " + ms
                        + " ms (render frame " + frame[0] + ")");
                return beat[0] < 6;
            };

            new Conductor()
                    .seat(render, 0)                // eager — vsync paces the loop
                    .seat(logic, 500_000_000L)      // every 500 ms
                    .run();

            window.drain();
        }
        System.out.println("[conductor] both players retired; clean exit");
    }
}
