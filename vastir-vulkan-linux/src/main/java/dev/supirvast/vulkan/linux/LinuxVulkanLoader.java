package dev.supirvast.vulkan.linux;

import dev.supirvast.vulkan.VulkanLoaderLibrary;

/** The Vulkan loader on Linux: {@code libvulkan.so.1}, the versioned soname: the bare libvulkan.so is a development symlink, present only where the -dev package is installed. */
public final class LinuxVulkanLoader implements VulkanLoaderLibrary {

    @Override
    public String fileName() {
        return "libvulkan.so.1";
    }
}
