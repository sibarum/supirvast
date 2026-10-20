package dev.supirvast.vulkan;

/**
 * Where the Vulkan loader is on this operating system: the one fact {@link VkLoader} needs that differs per OS. An
 * OS module ({@code vastir-vulkan-windows}, {@code vastir-vulkan-linux}) provides it through {@link java.util.ServiceLoader},
 * and the build puts exactly one of them on the class path, so this module holds no platform knowledge of its own.
 */
public interface VulkanLoaderLibrary {

    /** The loader's file name, as the OS's dynamic linker finds it: {@code vulkan-1.dll}, {@code libvulkan.so.1}. */
    String fileName();
}
