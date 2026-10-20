package dev.supirvast.vulkan.windows;

import dev.supirvast.vulkan.VulkanLoaderLibrary;

/** The Vulkan loader on Windows: {@code vulkan-1.dll}, the loader every Vulkan driver installs into System32. */
public final class WindowsVulkanLoader implements VulkanLoaderLibrary {

    @Override
    public String fileName() {
        return "vulkan-1.dll";
    }
}
