package dev.supirvast.vastir.preview;

import dev.supirvast.vastir.tools.Fullscreen;

/**
 * Phase 0 first light: open a Vulkan window that presents a fullscreen fragment shader authored in the
 * {@code core} IR. Both shaders come from {@link Fullscreen}; the whole path is core IR → SPIR-V → pipeline →
 * present, with no vertex buffer.
 *
 * <pre>java dev.supirvast.vastir.preview.FullscreenDemo [maxFrames]</pre>
 *
 * With no argument it renders until the window is closed; with a positive {@code maxFrames} it presents that many
 * frames and exits — a hands-off smoke test of the present path on real hardware.
 */
public final class FullscreenDemo {

    private FullscreenDemo() {
    }

    public static void main(String[] args) {
        int maxFrames = args.length > 0 ? Integer.parseInt(args[0]) : 0;
        byte[] vert = Fullscreen.triangleVertexSpirv();
        byte[] frag = Fullscreen.constantColorFragmentSpirv(0.11, 0.22, 0.44, 1.0);
        try (WindowedVulkanContext ctx = new WindowedVulkanContext(
                "SupirVast — fullscreen (Phase 0)", 960, 600,
                vert, Fullscreen.ENTRY_POINT, frag, Fullscreen.ENTRY_POINT)) {
            ctx.run(maxFrames);
        }
        System.out.println("[fullscreen-demo] presented " + (maxFrames > 0 ? maxFrames + " frame(s)" : "until close")
                + "; clean exit");
    }
}
